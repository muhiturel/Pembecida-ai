import os
import re
import json
import html as _html
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from difflib import SequenceMatcher

import requests
from lxml import etree
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# OpenAI opsiyonel: env yoksa fallback cevap döner
try:
    from openai import OpenAI
except Exception:
    OpenAI = None


# -----------------------------
# App / Config
# -----------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # widget farklı domainlerde çalışıyor
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEED_URL = os.environ.get("FEED_URL", "").strip()
STORE_PATH = "products.json"

UTM_SOURCE = "pembegpt"
UTM_MEDIUM = "chatbot"
UTM_CAMPAIGN = "pembecida"

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if (OpenAI and OPENAI_API_KEY) else None


# -----------------------------
# Models
# -----------------------------
class ChatIn(BaseModel):
    query: str
    page_url: str | None = None


# -----------------------------
# Text helpers
# -----------------------------
def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_html(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = _html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()


# -----------------------------
# Storage
# -----------------------------
def load_products() -> List[Dict[str, Any]]:
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f) or []
    except Exception:
        return []


def save_products(products: List[Dict[str, Any]]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)


# -----------------------------
# UTM
# -----------------------------
def add_utm(url: str) -> str:
    """UTM ekle. Var olan parametreleri bozmadan ekler/overwrite eder."""
    if not url:
        return url
    try:
        u = urlparse(url)
        q = dict(parse_qsl(u.query, keep_blank_values=True))
        q["utm_source"] = UTM_SOURCE
        q["utm_medium"] = UTM_MEDIUM
        q["utm_campaign"] = UTM_CAMPAIGN
        new_q = urlencode(q, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))
    except Exception:
        return url


# -----------------------------
# Price formatting
# -----------------------------
def format_price(p: Dict[str, Any]) -> str:
    cur = (p.get("currency") or "TL").strip() or "TL"
    candidates = [
        p.get("price_special_vat_included"),
        p.get("price_list_vat_included"),
        p.get("price_special"),
        p.get("price_list"),
    ]
    for c in candidates:
        try:
            if c is None:
                continue
            if isinstance(c, (int, float)):
                val = float(c)
            else:
                s = str(c).strip().replace(",", ".")
                if not s:
                    continue
                val = float(s)
            if val > 0:
                return f"{val:.2f} {cur}"
        except Exception:
            continue
    return ""


# -----------------------------
# Feed parsing (NEW XML)
# -----------------------------
def parse_products_from_xml(xml_bytes: bytes) -> List[Dict[str, Any]]:
    """
    Beklenen yapı:
      <product>
        <code>...</code>  (sıra no, büyük => yeni)
        <name><![CDATA[...]]></name>
        <product_link><![CDATA[https://...]]></product_link>
        <brand><![CDATA[Smiggle]]></brand>
        <category_path><![CDATA[Çanta > ...]]></category_path>
        <detail><![CDATA[<div>..</div>]]></detail>
        <seo_title>...</seo_title>
        <seo_description>...</seo_description>
        <images><img_item><![CDATA[...]]></img_item>...</images>
        <subproducts>
          <subproduct>
            <type1><![CDATA[Lila]]></type1>   # renk
          </subproduct>
        </subproducts>
      </product>
    """
    root = etree.fromstring(xml_bytes)
    out: List[Dict[str, Any]] = []

    for prod in root.findall(".//product"):
        code = safe_text(prod.find("code"))  # önemli: yenilik sırası
        name = safe_text(prod.find("name"))
        brand = safe_text(prod.find("brand"))
        link = safe_text(prod.find("product_link"))
        category_path = safe_text(prod.find("category_path"))
        model = safe_text(prod.find("model"))

        price_list = safe_text(prod.find("price_list"))
        price_list_vat_included = safe_text(prod.find("price_list_vat_included"))
        price_special = safe_text(prod.find("price_special"))
        price_special_vat_included = safe_text(prod.find("price_special_vat_included"))
        currency = safe_text(prod.find("currency"))
        stock = safe_text(prod.find("stock"))

        detail_html = safe_text(prod.find("detail"))
        seo_title = safe_text(prod.find("seo_title"))
        seo_description = safe_text(prod.find("seo_description"))

        images: List[str] = []
        for img in prod.findall(".//images/img_item"):
            t = safe_text(img)
            if t:
                images.append(t)

        colors: List[str] = []
        for sp in prod.findall(".//subproducts/subproduct"):
            c = safe_text(sp.find("type1"))
            if c:
                colors.append(c)

        # Minimum şartlar
        if not link or not name:
            continue

        # Arama metni
        detail_txt = strip_html(detail_html)
        search_text = norm(" ".join([
            name,
            brand,
            category_path,
            model,
            " ".join(colors),
            seo_title,
            seo_description,
            detail_txt
        ]))

        # code int
        try:
            code_i = int(re.sub(r"\D+", "", code) or "0")
        except Exception:
            code_i = 0

        p = {
            "id": code or link,              # stabil id
            "code": code_i,                  # yenilik sırası
            "name": name,
            "brand": brand,
            "brand_name": f"{brand} {name}".strip(),
            "product_link": link,
            "category_path": category_path,
            "model": model,
            "currency": currency or "TL",
            "stock": stock,
            "price_list": price_list,
            "price_list_vat_included": price_list_vat_included,
            "price_special": price_special,
            "price_special_vat_included": price_special_vat_included,
            "images": images,
            "colors": list(dict.fromkeys([c.strip() for c in colors if c.strip()])),
            "seo_title": seo_title,
            "seo_description": seo_description,
            "detail_text": detail_txt,
            "search_text": search_text,
        }
        out.append(p)

    # “en yeni” = code büyük
    out.sort(key=lambda x: x.get("code", 0), reverse=True)
    return out


def fetch_and_index_feed() -> int:
    if not FEED_URL:
        save_products([])
        return 0

    r = requests.get(FEED_URL, timeout=120)
    r.raise_for_status()
    products = parse_products_from_xml(r.content)
    save_products(products)
    return len(products)


# -----------------------------
# Search
# -----------------------------
COLOR_HINTS: Dict[str, List[str]] = {
    "pembe": ["pembe", "pink", "fuşya", "fuchsia"],
    "siyah": ["siyah", "black"],
    "beyaz": ["beyaz", "white"],
    "mavi": ["mavi", "blue", "lacivert", "navy"],
    "mor": ["mor", "purple", "lila", "lavanta"],
    "kırmızı": ["kırmızı", "kirmizi", "red"],
    "yeşil": ["yeşil", "yesil", "green"],
    "turuncu": ["turuncu", "orange"],
    "sarı": ["sarı", "sari", "yellow"],
}

BAG_KEYWORDS = ["çanta", "canta", "cantası", "cantasi", "sırt", "sirt", "okul", "çekçek", "cekcek", "trolley", "valiz", "beslenme"]

BRAND_KNOWN = ["smiggle", "pop mart", "popmart", "pembecida"]

KEYWORDS_KNOWN = ["crybaby", "labubu", "skullpanda", "hirono", "dimoo", "molly"]


def fix_typo_token(token: str, candidates: List[str], min_ratio: float) -> Tuple[str, bool]:
    best_ratio = 0.0
    best = token
    for c in candidates:
        r = SequenceMatcher(None, token, c).ratio()
        if r > best_ratio:
            best_ratio = r
            best = c
    if best != token and best_ratio >= min_ratio:
        return best, True
    return token, False


def normalize_query(q: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    Konservatif typo düzeltme:
      smgle/smigle -> smiggle
      cyrbaby -> crybaby
    """
    nq = norm(q)
    if not nq:
        return q, []

    tokens = nq.split()
    changes: List[Tuple[str, str]] = []

    out = []
    for t in tokens:
        t2 = t

        # marka adayları
        t2, changed = fix_typo_token(t2, [norm(x) for x in BRAND_KNOWN], min_ratio=0.84)
        if changed:
            changes.append((t, t2))

        # keyword adayları
        t3, changed2 = fix_typo_token(t2, [norm(x) for x in KEYWORDS_KNOWN], min_ratio=0.84)
        if changed2 and t3 != t2:
            changes.append((t2, t3))
            t2 = t3

        out.append(t2)

    return " ".join(out), changes


def detect_color_terms(nq: str) -> List[str]:
    terms: List[str] = []
    for _, arr in COLOR_HINTS.items():
        for t in arr:
            if norm(t) in nq:
                terms.append(norm(t))
    return list(dict.fromkeys(terms))


def has_bag_intent(nq: str) -> bool:
    return any(norm(k) in nq for k in BAG_KEYWORDS)


def matches_color(p: Dict[str, Any], color_terms: List[str]) -> bool:
    if not color_terms:
        return True
    p_colors = norm(" ".join(p.get("colors", []) or []))
    p_name = norm(p.get("brand_name", ""))
    return any(ct in p_colors or ct in p_name for ct in color_terms)


def matches_bag(p: Dict[str, Any]) -> bool:
    hay = norm(" ".join([
        p.get("name", ""),
        p.get("brand", ""),
        p.get("category_path", ""),
        p.get("model", ""),
    ]))
    return any(k in hay for k in [norm(x) for x in BAG_KEYWORDS])


def search_products(products: List[Dict[str, Any]], q: str, k: int = 6) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    nq0 = norm(q)
    nq, changes = normalize_query(q)
    nq = norm(nq)

    meta = {
        "used_query": nq,
        "typo": changes,
        "color_terms": [],
        "strict_color": False,
        "bag_intent": False,
    }

    if not nq:
        return [], meta

    color_terms = detect_color_terms(nq)
    bag_int = has_bag_intent(nq)

    meta["color_terms"] = color_terms
    meta["strict_color"] = True if color_terms else False
    meta["bag_intent"] = bag_int

    tokens = nq.split()
    
    scored: List[Tuple[int, Dict[str, Any]]] = []
    for p in products:
        # Query içinde marka geçiyorsa, diğer markaları ele (çok daha stabil)
        q_has_smiggle = ("smiggle" in nq)
        q_has_popmart = ("pop mart" in nq) or ("popmart" in nq)

        b = norm(p.get("brand", ""))
        if q_has_smiggle and ("smiggle" not in b):
            continue
        if q_has_popmart and (("pop mart" not in b) and ("popmart" not in b)):
            continue
        
        # Hard filter: renk varsa renge uymayanı ele
        if color_terms and not matches_color(p, color_terms):
            continue
        # Hard filter: çanta niyeti varsa çanta olmayanı ele
        if bag_int and not matches_bag(p):
            continue

        hay = p.get("search_text", "")
        if not hay:
            continue

        score = 0

        # token eşleşmesi
        for t in tokens:
            if t and t in hay:
                score += 2

        # brand boost
        b = norm(p.get("brand", ""))
        if b and b in nq:
            score += 3

        # renk boost (zaten filtreledi ama sıralama için)
        if color_terms:
            p_colors = norm(" ".join(p.get("colors", []) or []))
            score += 4 * sum(1 for ct in color_terms if ct in p_colors or ct in hay)

        # yenilik boost (code büyük = yeni)
        score += 1 if p.get("code", 0) > 0 else 0

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: (x[0], x[1].get("code", 0)), reverse=True)
    hits = [p for _, p in scored[:k]]

    # Eğer renkli arama yaptı ve hiç yoksa: “bulamadım” dönmek doğru (alakasız göstermeyelim)
    return hits, meta


# -----------------------------
# LLM answer cleanup (kartlar ayrı gösterilecek)
# -----------------------------
def clean_llm_answer(text: str) -> str:
    """
    LLM bazen ürün linklerini markdown liste olarak döküyor.
    Kartlar zaten ayrı basıldığı için ürün liste satırlarını temizler.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        # Markdown ürün listelerini temizle
        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and ("[" in s and "](" in s):
            if s.startswith("-") or s.startswith("["):
                continue
        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and (s.startswith("-") or s.startswith("http")):
            continue
        out.append(ln)

    cleaned = "\n".join(out).strip()
    if len(cleaned) < 20:
        cleaned = "Size uygun birkaç öneri çıkardım; aşağıda kartlar halinde görebilirsiniz. İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daraltayım."
    return cleaned


# -----------------------------
# API
# -----------------------------
@app.get("/pb-chat/health")
def health():
    return {"ok": True, "feed": bool(FEED_URL)}


@app.get("/pb-chat/reindex")
def reindex_get():
    count = fetch_and_index_feed()
    return {"ok": True, "count": count}


@app.post("/pb-chat/reindex")
def reindex_post():
    count = fetch_and_index_feed()
    return {"ok": True, "count": count}


@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = load_products()
    rows = []
    for p in products[:limit]:
        rows.append({
            "id": p.get("id"),
            "code": p.get("code"),
            "brand": p.get("brand"),
            "name": p.get("name"),
            "colors": p.get("colors"),
            "category_path": p.get("category_path"),
            "product_link": p.get("product_link"),
        })
    return {"count": len(products), "rows": rows}


@app.get("/pb-chat/debug/search")
def debug_search(q: str, k: int = 10):
    products = load_products()
    hits, meta = search_products(products, q, k=k)
    # küçük cevap
    return {
        "query": q,
        "meta": meta,
        "hits": [{
            "code": h.get("code"),
            "brand": h.get("brand"),
            "name": h.get("name"),
            "colors": h.get("colors"),
            "link": h.get("product_link"),
        } for h in hits]
    }


@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()

    if not products:
        return {
            "answer": "Şu an ürün listesini yükleyemedim. Lütfen biraz sonra tekrar dener misiniz?",
            "products": []
        }

    hits, meta = search_products(products, inp.query, k=6)

    if not hits:
        return {
            "answer": (
                'Bu aramada eşleşen ürün bulamadım. '
                'İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). '
                f'İade/kargo gibi konular için <a href="{add_utm("https://www.pembecida.com/sikca-sorulan-sorular")}">Sıkça Sorulan Sorular</a> sayfamızı inceleyebilirsiniz.'
            ),
            "products": [],
            "meta": meta
        }

    # UI kartları
    ui_products = []
    for p in hits[:5]:
        url = add_utm(p.get("product_link") or "")
        price = format_price(p)
        img = (p.get("images") or [""])[0] if (p.get("images") or []) else ""
        ui_products.append({
            "title": p.get("brand_name") or p.get("name") or "",
            "link": url,
            "price": price,
            "image": img,
        })

    # OpenAI yoksa bile UI çalışsın
    if not client:
        return {
            "answer": "Size uygun birkaç öneri hazırladım; aşağıda kartlar halinde görebilirsiniz. İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daha da daraltayım.",
            "products": ui_products,
            "meta": meta
        }

    # LLM'e sadece bulunan ürünleri veriyoruz (hallucination engeli)
    safe_products = [{
        "title": u["title"],
        "price": u["price"],
        "link": u["link"],
    } for u in ui_products]

    system = (
        "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
        "Kısa cevap ver. "
        "İade/kargo vb. sorularda SSS sayfasına yönlendir: https://www.pembecida.com/sikca-sorulan-sorular . "
        "ÇOK ÖNEMLİ: Sana verilen ürün listesinde olmayan hiçbir ürünü ASLA önerme; isim veya link UYDURMA. "
        "ÇIKTI KURALI: Sadece 2-3 cümlelik kısa bir karşılama ve yönlendirme yaz. "
        "Ürünleri metin olarak listeleme; ürünler kart olarak gösterilecek."
    )

    user_payload = {
        "query": inp.query,
        "products": safe_products
    }

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
    )

    answer = clean_llm_answer(resp.output_text)

    return {
        "answer": answer,
        "products": ui_products,
        "meta": meta
    }


# -----------------------------
# Widget JS (served from API)
# -----------------------------
@app.get("/pb-chat/widget.js")
def widget():
    js = r"""
(() => {
  const API_BASE = "https://pembecida-ai.onrender.com";

  const style = document.createElement("style");
  style.innerHTML = `
    #pb_msgs a { color:#0645AD; text-decoration:underline; }
    #pb_msgs a:visited { color:#0b0080; }

    .pb-gpt-wrap{
      position:fixed;
      right:16px;
      bottom:16px;
      z-index:999999;
    }

    .pb-gpt-wrap::before{
      content:"";
      position:absolute;
      inset:-7px;
      border-radius:999px;
      background: linear-gradient(45deg, #ff5db1, #ff7a00, #ff5db1);
      filter: blur(7px);
      opacity: .95;
      animation: pbGlow 2.6s linear infinite;
      z-index:1;
      pointer-events:none;
    }

    @keyframes pbGlow{
      0%{ transform: rotate(0deg); }
      100%{ transform: rotate(360deg); }
    }

    .pb-gpt-btn{
      position:relative;
      z-index:2;
      padding:20px 28px;
      font-size:18px;
      border-radius:999px;
      border:0;
      cursor:pointer;
      background: linear-gradient(45deg, #ff5db1, #ff7a00);
      color:#fff;
      font-weight:800;
      box-shadow: 0 10px 22px rgba(0,0,0,.18);
      -webkit-tap-highlight-color: transparent;
    }

    @media (max-width: 480px){
      .pb-gpt-wrap{
        left:50%;
        right:auto;
        transform: translateX(-50%) scale(0.85);
        transform-origin: center bottom;
      }
    }
  `;
  document.head.appendChild(style);

  const btnWrap = document.createElement("div");
  btnWrap.className = "pb-gpt-wrap";

  const btn = document.createElement("button");
  btn.className = "pb-gpt-btn";
  btn.innerText = "PembeGPT";

  btnWrap.appendChild(btn);
  document.body.appendChild(btnWrap);

  const box = document.createElement("div");
  box.style.cssText = "display:none;position:fixed;right:16px;bottom:64px;z-index:999999;width:340px;max-width:calc(100vw - 32px);height:520px;background:#fff;border:2px solid #9EA3A8;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;font-family:system-ui;";
  box.innerHTML = `
    <div style="padding:12px 14px;border-bottom:1px solid #eee;font-weight:600;">Pembecida Asistan</div>
    <div id="pb_msgs" style="padding:10px 14px;height:390px;overflow:auto;font-size:14px;line-height:1.35;"></div>
    <div style="display:flex;gap:8px;padding:10px 10px;border-top:1px solid #eee;">
      <input id="pb_in" placeholder="Ne arıyorsunuz? (örn: 8 yaş hediye, Pop Mart, kalem kutusu...)" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:10px;"/>
      <button id="pb_send" style="padding:10px 12px;border-radius:10px;border:0;cursor:pointer;">Gönder</button>
    </div>
  `;
  document.body.appendChild(box);

  const msgs = box.querySelector("#pb_msgs");
  const input = box.querySelector("#pb_in");
  const send = box.querySelector("#pb_send");

  const addMsg = (who, text) => {
    const d = document.createElement("div");
    d.style.margin = "8px 0";
    d.innerHTML = `<div style="font-weight:600;margin-bottom:2px;">${who}</div><div>${text}</div>`;
    msgs.appendChild(d);
    msgs.scrollTop = msgs.scrollHeight;
  };

  btn.onclick = () => {
    box.style.display = box.style.display === "none" ? "block" : "none";
    if (msgs.childElementCount === 0) {
      addMsg("Pembecida", "Merhaba! Size yardımcı olayım: Ne arıyorsunuz? Örn: “Smiggle kalem kutusu”, “8 yaş hediye”, “Pop Mart blind box”.");
    }
  };

  const escapeHtml = (s) => (s || "").replace(/</g,"&lt;").replace(/>/g,"&gt;");

  const renderProducts = (products) => {
    if (!products || !products.length) return "";
    return products.map(p => {
      const title = escapeHtml(p.title || "");
      const price = escapeHtml(p.price || "");
      const link = p.link || "#";
      const img = p.image || "";
      return `
        <div style="display:flex;gap:10px;padding:10px;border:1px solid #eee;border-radius:12px;margin-top:10px;">
          ${img ? `<img src="${img}" alt="${title}" style="width:64px;height:64px;object-fit:cover;border-radius:10px;border:1px solid #eee;">` : ``}
          <div style="flex:1;min-width:0;">
            <div style="font-weight:600;font-size:14px;line-height:1.25;margin-bottom:4px;">${title}</div>
            ${price ? `<div style="font-size:13px;color:#333;margin-bottom:8px;">${price}</div>` : ``}
            <a href="${link}" target="_self" style="display:inline-block;text-decoration:none;padding:8px 10px;border-radius:10px;border:1px solid #ddd;font-size:13px;">
              Ürünü İncele
            </a>
          </div>
        </div>
      `;
    }).join("");
  };

  const doSend = async () => {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    addMsg("Siz", q);
    addMsg("Pembecida", "Hemen bakıyorum…");

    try {
      const res = await fetch(API_BASE + "/pb-chat/chat", {
        method: "POST",
        headers: {"Content-Type":"application/json"},
        body: JSON.stringify({ query: q, page_url: location.href })
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error((data && (data.detail || data.error)) || ("HTTP " + res.status));

      const answerHtml = (data.answer || "").replace(/\n/g, "<br/>");
      const cardsHtml = renderProducts(data.products || []);
      msgs.lastChild.querySelector("div:last-child").innerHTML = answerHtml + cardsHtml;

    } catch (e) {
      msgs.lastChild.querySelector("div:last-child").innerHTML =
        "Şu an bağlantıda bir sorun yaşadım. Lütfen sayfayı yenileyip tekrar dener misiniz? (Hata: " + (e.message || e) + ")";
    }
  };

  send.onclick = doSend;
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") doSend(); });
})();
""".strip()
    return Response(js, media_type="application/javascript; charset=utf-8")

