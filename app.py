import os, re, json
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

import requests
from lxml import etree
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from openai import OpenAI


# =========================
# App / Config
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEED_URL = os.environ.get("FEED_URL", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "products.json")

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


class ChatIn(BaseModel):
    query: str
    page_url: Optional[str] = None


# =========================
# Helpers
# =========================
def is_irrelevant_query(q: str) -> bool:
    """
    Ürün araması olmayan sorguları yakalamak için basit bir gate.
    Amaç: alakasızsa arama/LLM çalıştırmadan net yönlendirmek.
    """
    nq = norm(q)
    if not nq:
        return True

    # Çok kısa tek kelime ve ürün/marka sinyali yoksa
    product_signals = [
        "smiggle", "pop", "mart", "labubu", "crybaby", "skullpanda", "dimoo",
        "çanta", "canta", "kalem", "kalem kutusu", "kalemkutusu", "beslenme",
        "suluk", "termos", "oyuncak", "figur", "figür", "blind box", "sürpriz", "surpriz"
    ]
    if len(nq) <= 3 and not any(s in nq for s in product_signals):
        return True

    # Genel alakasız “konu” kelimeleri
    off_topic = [
        "galatasaray", "fenerbahçe", "beşiktaş", "trabzon",
        "hava durumu", "deprem", "seçim", "politika",
        "dolar", "euro", "borsa", "bitcoin",
        "kimdir", "nedir", "nasıl yapılır", "tarif", "reçete", "film", "dizi"
    ]
    if any(t in nq for t in off_topic):
        # Ama ürün sinyali varsa engelleme (örn "galatasaray çantası" ürün olabilir)
        if not any(s in nq for s in product_signals):
            return True

    return False

def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s

def safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(str(x).strip())
    except Exception:
        return default

def add_utm(url: str) -> str:
    if not url:
        return url
    try:
        u = urlparse(url)
        q = dict(parse_qsl(u.query, keep_blank_values=True))
        q.setdefault("utm_source", "pembegpt")
        q.setdefault("utm_medium", "chatbot")
        q.setdefault("utm_campaign", "pembecida")
        new_q = urlencode(q, doseq=True)
        return urlunparse((u.scheme, u.netloc, u.path, u.params, new_q, u.fragment))
    except Exception:
        return url

def load_products() -> List[Dict[str, Any]]:
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []

def save_products(products: List[Dict[str, Any]]) -> None:
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)

def pick_text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    return (el.text or "").strip()

def first_text(node: etree._Element, tags: List[str], ns: Optional[Dict[str, str]] = None) -> str:
    for t in tags:
        try:
            if ":" in t and ns:
                prefix, local = t.split(":", 1)
                el = node.find(f".//{prefix}:{local}", namespaces=ns)
            else:
                el = node.find(f".//{t}")
            v = pick_text(el)
            if v:
                return v
        except Exception:
            continue
    return ""

def all_texts(node: etree._Element, tag: str, ns: Optional[Dict[str, str]] = None) -> List[str]:
    out = []
    try:
        if ":" in tag and ns:
            prefix, local = tag.split(":", 1)
            els = node.findall(f".//{prefix}:{local}", namespaces=ns)
        else:
            els = node.findall(f".//{tag}")
        for el in els:
            v = pick_text(el)
            if v:
                out.append(v)
    except Exception:
        pass
    return out

def looks_like_url(s: str) -> bool:
    s = (s or "").strip()
    return s.startswith("http://") or s.startswith("https://")

def find_first_image_anywhere(node: etree._Element) -> str:
    """
    Yeni XML'de görseller farklı adlarla gelebiliyor.
    Bu fonksiyon product node altındaki tüm child’ları dolaşır,
    'img/image/picture' vb. isimlerde VEYA içinde 'img'/'image' geçen tag’lerde
    http ile başlayan ilk URL'i bulur.
    """
    # önce bilinen aday tagler
    candidate_tags = [
        "image", "img", "image_link", "imageurl", "image_url", "picture", "pic",
        "Images", "imgs", "Imgs", "images", "Pictures", "pictures",
    ]
    for t in candidate_tags:
        v = first_text(node, [t])
        if looks_like_url(v):
            return v

    # nested image lists
    for t in ["img", "image", "image_link", "picture", "pic"]:
        vals = all_texts(node, t)
        for v in vals:
            if looks_like_url(v):
                return v

    # full scan (en sağlam)
    try:
        for el in node.iter():
            tag = str(getattr(el, "tag", "")).lower()
            txt = (el.text or "").strip()
            if not txt:
                continue
            if not looks_like_url(txt):
                continue
            if ("img" in tag) or ("image" in tag) or ("pic" in tag) or tag in ["url", "src"]:
                return txt
    except Exception:
        pass

    return ""


# =========================
# Category / Intent logic
# =========================
CATEGORY_INTENTS = {
    "sirt_okul_cantasi": ["sırt çantası", "sirt cantasi", "okul çantası", "okul cantasi", "backpack", "school bag"],
    "cekcekli": ["çekçekli", "cekcekli", "çekçek", "cekcek", "trolley", "valizli", "kabin boy"],
    "beslenme": ["beslenme çantası", "beslenme cantasi", "lunch", "lunchbag"],
    "kalem_kutusu": ["kalem kutusu", "kalemkutusu", "pencil case", "kalemlik"],
    "suluk_termos": ["suluk", "termos", "matara", "bottle", "thermos"],
    "blind_box": ["blind box", "sürpriz kutu", "surpriz kutu"],
    "oyuncak_figur": ["oyuncak", "figür", "figur", "koleksiyon", "collectible"],
    "labubu": ["labubu"],
    "crybaby": ["crybaby", "cry baby"],
    "skullpanda": ["skullpanda", "skull panda"],
    "hacipupu": ["hacipupu"],
    "dimoo": ["dimoo"],
}

INTENT_TO_CATEGORY_PATH_HINTS = {
    "sirt_okul_cantasi": ["sırt", "okul", "canta", "çanta", "backpack"],
    "cekcekli": ["çekçek", "cekcek", "trolley", "valiz", "kabin"],
    "beslenme": ["beslenme"],
    "kalem_kutusu": ["kalem kutusu", "kalemkutusu", "kalem-kutusu"],
    "suluk_termos": ["suluk", "termos", "matara"],
    "blind_box": ["blind", "sürpriz", "surpriz"],
    "oyuncak_figur": ["oyuncak", "figür", "figur", "koleksiyon"],
    "labubu": ["labubu"],
    "crybaby": ["crybaby"],
    "skullpanda": ["skullpanda"],
    "hacipupu": ["hacipupu"],
    "dimoo": ["dimoo"],
}

COLOR_TERMS = {
    "pembe": ["pembe", "pink"],
    "siyah": ["siyah", "black"],
    "mavi": ["mavi", "blue"],
    "mor": ["mor", "purple"],
    "kırmızı": ["kırmızı", "kirmizi", "red"],
    "yeşil": ["yeşil", "yesil", "green"],
    "turuncu": ["turuncu", "orange"],
    "sarı": ["sarı", "sari", "yellow"],
    "beyaz": ["beyaz", "white"],
    "gri": ["gri", "gray", "grey"],
    "çok renkli": ["çok renkli", "cok renkli", "multicolor"],
}

def detect_intents(nq: str) -> List[str]:
    intents = []
    for ik, terms in CATEGORY_INTENTS.items():
        if any(norm(t) in nq for t in terms):
            intents.append(ik)
    return intents

def detect_colors(nq: str) -> List[str]:
    cols = []
    for cname, terms in COLOR_TERMS.items():
        if any(norm(t) in nq for t in terms):
            cols.append(cname)
    return cols

def detect_brand(nq: str) -> Optional[str]:
    if "smiggle" in nq:
        return "smiggle"
    if "pop mart" in nq or "popmart" in nq:
        return "pop mart"
    return None

def conservative_typo_fix(q: str) -> str:
    nq = norm(q)
    if not nq:
        return q
    tokens = nq.split()
    if len(tokens) > 5:
        return q

    known = [
        "smiggle", "crybaby", "labubu", "skullpanda", "hacipupu", "dimoo", "popmart", "pop", "mart"
    ]
    new_tokens = []
    changed = False
    for t in tokens:
        best_k = None
        best_r = 0.0
        for k in known:
            r = SequenceMatcher(None, t, k).ratio()
            if r > best_r:
                best_r = r
                best_k = k
        if best_k and best_r >= 0.84 and t != best_k:
            new_tokens.append(best_k)
            changed = True
        else:
            new_tokens.append(t)

    if not changed:
        return q

    fixed = " ".join(new_tokens).replace("popmart", "pop mart")
    return fixed


# =========================
# Feed indexing (robust + images)
# =========================
def fetch_and_index_feed() -> int:
    if not FEED_URL:
        raise RuntimeError("FEED_URL env is missing")

    r = requests.get(FEED_URL, timeout=140)
    r.raise_for_status()
    xml = etree.fromstring(r.content)

    # A) Google Shopping style <item> + g:
    items = xml.findall(".//item")
    if items:
        ns = {"g": "http://base.google.com/ns/1.0"}
        products: List[Dict[str, Any]] = []
        for it in items:
            def gget(tag: str) -> str:
                return first_text(it, [f"g:{tag}", tag], ns=ns)

            pid = gget("id")
            title = gget("title")
            brand = gget("brand")
            link = gget("link")
            price = gget("price")
            sale_price = gget("sale_price")
            product_type = gget("product_type")
            image = gget("image_link") or find_first_image_anywhere(it)

            if not pid or not link:
                continue

            products.append({
                "id": str(pid).strip(),
                "code": safe_int(pid, 0),
                "brand": brand,
                "name": title,
                "category_path": product_type or "",
                "colors": [],
                "price": sale_price or price or "",
                "product_link": link,
                "link": add_utm(link),
                "image": image,
                "image_link": image,
                "search_text": " ".join([brand, title, product_type, link]).strip(),
            })

        products.sort(key=lambda x: x.get("code", 0), reverse=True)
        save_products(products)
        return len(products)

    # B) Custom feed: product-like nodes
    candidates = xml.findall(".//product") + xml.findall(".//Product") + xml.findall(".//urun") + xml.findall(".//Urun")
    if not candidates:
        candidates = [n for n in xml.iter() if n.find(".//product_link") is not None or n.find(".//link") is not None]

    products: List[Dict[str, Any]] = []

    for p in candidates:
        pid = first_text(p, ["id", "ID", "product_id", "urunid", "webserviskodu", "code", "Kod"])
        code = first_text(p, ["code", "Code", "sira", "siraNo", "sirano"])

        brand = first_text(p, ["brand", "Brand", "marka", "Marka"])
        name = first_text(p, ["name", "Name", "title", "Title", "urunadi", "UrunAdi"])
        link = first_text(p, ["product_link", "ProductLink", "link", "Link", "url", "URL"])
        category_path = first_text(p, ["category_path", "CategoryPath", "category", "Category", "kategori", "Kategori"])
        price = first_text(p, ["price", "Price", "fiyat", "Fiyat", "sale_price", "SalePrice"])

        # colors: subproducts/type1 + direct type1
        colors = []
        for sp in p.findall(".//subproducts") + p.findall(".//Subproducts") + p.findall(".//subproduct") + p.findall(".//Subproduct"):
            c = first_text(sp, ["type1", "Type1", "color", "Color", "renk", "Renk"])
            if c:
                colors.append(c)
        c2 = first_text(p, ["type1", "Type1", "color", "Color", "renk", "Renk"])
        if c2:
            colors.append(c2)
        colors = list(dict.fromkeys([x.strip() for x in colors if x.strip()]))

        image = find_first_image_anywhere(p)

        if not pid:
            pid = link or name
        if not link:
            continue

        code_i = safe_int(code, safe_int(pid, 0))

        products.append({
            "id": str(pid).strip(),
            "code": code_i,
            "brand": brand,
            "name": name,
            "category_path": category_path,
            "colors": colors,
            "price": price or "",
            "product_link": link,
            "link": add_utm(link),
            "image": image,
            "image_link": image,
            "search_text": " ".join([brand, name, category_path, " ".join(colors), link]).strip(),
        })

    products.sort(key=lambda x: x.get("code", 0), reverse=True)
    save_products(products)
    return len(products)


# =========================
# Search (NO irrelevant hits)
# =========================
def product_haystack(p: Dict[str, Any]) -> str:
    return norm(" ".join([
        p.get("brand", "") or "",
        p.get("name", "") or "",
        p.get("category_path", "") or "",
        " ".join(p.get("colors", []) or []),
        p.get("product_link", "") or "",
    ]))

def search_products(products: List[Dict[str, Any]], q: str, k: int = 6) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    original_q = q or ""
    q2 = conservative_typo_fix(original_q)
    nq = norm(q2)

    meta = {
        "query_original": original_q,
        "query_fixed": q2,
        "intent": [],
        "colors": [],
        "brand": None,
    }

    if not nq:
        return [], meta

    intents = detect_intents(nq)
    colors = detect_colors(nq)
    brand = detect_brand(nq)

    meta["intent"] = intents
    meta["colors"] = colors
    meta["brand"] = brand

    tokens = [t for t in nq.split() if t]

    scored: List[Tuple[int, int, Dict[str, Any]]] = []

    for p in products:
        hay = product_haystack(p)
        code = safe_int(p.get("code", 0), 0)

        # BRAND filter
        if brand == "smiggle" and "smiggle" not in norm(p.get("brand", "")):
            continue
        if brand == "pop mart":
            b = norm(p.get("brand", ""))
            if ("pop" not in b) and ("mart" not in b):
                continue

        # INTENT hard filter
        if intents:
            cat = norm(p.get("category_path", "") or "")
            name = norm(p.get("name", "") or "")
            link = norm(p.get("product_link", "") or "")
            ok_any = False
            for ix in intents:
                hints = INTENT_TO_CATEGORY_PATH_HINTS.get(ix, [])
                if any(h in cat for h in hints) or any(h in name for h in hints) or any(h in link for h in hints):
                    ok_any = True
                    break
            if not ok_any:
                continue

        # COLOR filter
        if colors:
            pcols = " ".join([norm(c) for c in (p.get("colors") or [])])
            pname = norm(p.get("name", "") or "")
            if not any((c in pcols) or (c in pname) for c in colors):
                continue

        # ===== base match =====
        base = sum(1 for t in tokens if t in hay)

        # Eğer hiçbir base/intent/brand/color tutmuyorsa -> ASLA skor verme (galatasaray bug fix)
        if base == 0 and not intents and not colors and not brand:
            continue

        score = 0
        score += base

        # boosts
        if intents:
            score += 4
        if colors:
            score += 3
        if brand:
            score += 2

        # recency boost sadece zaten bir şekilde eşleşme varsa
        if score > 0:
            score += min(2, code // 400)

        if score > 0:
            scored.append((score, code, p))

    scored.sort(key=lambda x: (x[0], x[1]), reverse=True)
    hits = [p for _, _, p in scored[:k]]
    return hits, meta


# =========================
# API Endpoints
# =========================
@app.get("/pb-chat/health")
def health():
    return {"ok": True, "feed": bool(FEED_URL)}

@app.get("/pb-chat/reindex")
@app.post("/pb-chat/reindex")
def reindex():
    cnt = fetch_and_index_feed()
    return {"ok": True, "count": cnt}

@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = load_products()
    return {"count": len(products), "rows": products[: max(0, limit)]}

@app.get("/pb-chat/debug/search")
def debug_search(q: str, k: int = 10):
    products = load_products()
    hits, meta = search_products(products, q, k=k)
    slim = []
    for p in hits:
        slim.append({
            "id": p.get("id"),
            "code": p.get("code"),
            "brand": p.get("brand"),
            "name": p.get("name"),
            "colors": p.get("colors", []),
            "category_path": p.get("category_path", ""),
            "product_link": p.get("product_link"),
            "image": p.get("image"),
        })
    return {"meta": meta, "hits": slim}

def to_widget_products(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in hits:
        out.append({
            "title": p.get("name") or "",
            "price": p.get("price") or "",
            "link": p.get("link") or add_utm(p.get("product_link") or ""),
            "image": p.get("image") or p.get("image_link") or "",
        })
    return out

@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()
    hits, meta = search_products(products, inp.query, k=6)

    # ✅ Alakasız sorgu kalibrasyonu: hiç arama/LLM çalıştırma
    if is_irrelevant_query(inp.query):
        return {
            "answer": (
                "Ben Pembecida’da ürün bulmanıza yardımcı olan bir alışveriş asistanıyım 😊<br/>"
                "Ne aradığınızı ürün tipi/marka ile yazarsanız hemen önerebilirim.<br/><br/>"
                "Örnekler:<br/>"
                "• “Smiggle termos”<br/>"
                "• “pembe çanta”<br/>"
                "• “8 yaş hediye Pop Mart”<br/>"
                "• “kalem kutusu”"
            ),
            "products": [],
            "meta": {"calibration": "irrelevant_query"}
        }
    
    # no hits => net ve uydurmayan yanıt
    if not hits:
        sss_url = "https://www.pembecida.com/sikca-sorulan-sorular"
        answer = (
            "Bu aramada eşleşen ürün bulamadım. "
            "İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). "
            f"İade/kargo gibi konular için <a href=\"{sss_url}\">Sıkça Sorulan Sorular</a> sayfamızı inceleyebilirsiniz."
        )
        return {"answer": answer, "products": [], "meta": meta}

    # LLM yoksa fallback
    if client is None:
        return {
            "answer": "Sizin için uygun seçenekler buldum. Aşağıdaki ürün kartlarından inceleyebilirsiniz.",
            "products": to_widget_products(hits),
            "meta": meta,
        }

    # ✅ Anti-hallucination: sadece arama sonuçlarına dayan, ürün uydurma
    system = (
        "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
        "SADECE sana verilen arama sonuçlarına dayanarak cevap ver. "
        "Arama sonuçları dışına çıkıp ürün/özellik uydurma, genelleme yapma. "
        "Ürünleri metin içinde listeleme; metin sadece kısa bir karşılama + yönlendirme olsun. "
        "Kullanıcı alakasız bir şey sorarsa 'bu konuda ürün bulamadım' diye net söyle. "
        "Fiyatlar değişmiş olabilir; en güncel bilgi ürün sayfasındadır. "
        "İade/kargo vb. sorularda SSS: https://www.pembecida.com/sikca-sorulan-sorular"
    )

    # Burada ürünleri METİN içine vermiyoruz (liste geri gelmesin diye).
    # Sadece niyet bilgisi veriyoruz.
    intent_hint = ""
    if meta.get("brand"):
        intent_hint += f"Marka: {meta['brand']}. "
    if meta.get("intent"):
        intent_hint += f"Kategori niyeti: {', '.join(meta['intent'])}. "
    if meta.get("colors"):
        intent_hint += f"Renk niyeti: {', '.join(meta['colors'])}. "

    user = (
        f"Kullanıcı sorgusu: {inp.query}\n"
        f"Düzeltilmiş sorgu: {meta.get('query_fixed')}\n"
        f"{intent_hint}\n"
        "Kısa ve net 1-2 cümleyle cevap ver. "
        "Cevabın sonunda mutlaka: 'Aşağıdaki ürün kartlarından inceleyebilirsiniz.' yaz."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )

    return {"answer": (resp.output_text or "").strip(), "products": to_widget_products(hits), "meta": meta}


# =========================
# Widget JS (senin mevcut widget mantığına uyumlu)
# =========================
@app.get("/pb-chat/widget.js")
def widget_js():
    js = r"""
(() => {
  const API_BASE = "https://pembecida-ai.onrender.com";

  const style = document.createElement("style");
  style.innerHTML = `
    #pb_msgs a { color:#0645AD; text-decoration:underline; }
    #pb_msgs a:visited { color:#0b0080; }

    .pb-gpt-wrap{ position:fixed; right:16px; bottom:16px; z-index:99999; }
    .pb-gpt-wrap::before{
      content:""; position:absolute; inset:-7px; border-radius:999px;
      background: linear-gradient(45deg, #ff5db1, #ff7a00, #ff5db1);
      filter: blur(7px); opacity:.95; animation: pbGlow 2.6s linear infinite;
      z-index:1; pointer-events:none;
    }
    @keyframes pbGlow{ 0%{transform:rotate(0)} 100%{transform:rotate(360deg)} }

    .pb-gpt-btn{
      position:relative; z-index:2;
      padding:20px 28px; font-size:18px;
      border-radius:999px; border:0; cursor:pointer;
      background: linear-gradient(45deg, #ff5db1, #ff7a00);
      color:#fff; font-weight:800;
      box-shadow:0 10px 22px rgba(0,0,0,.18);
      -webkit-tap-highlight-color: transparent;
    }

    @media (max-width: 480px){
      .pb-gpt-wrap{
        left:50%; right:auto;
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
  box.style.cssText =
    "display:none;position:fixed;right:16px;bottom:64px;z-index:99999;width:340px;max-width:calc(100vw - 32px);height:520px;background:#fff;border:2px solid #9EA3A8;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;font-family:system-ui;";
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
      addMsg("Pembecida", "Merhaba! Size yardımcı olayım: Ne arıyorsunuz? Örn: “Smiggle termos”, “pembe çanta”, “Pop Mart blind box”.");
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
            <a href="${link}" style="display:inline-block;text-decoration:none;padding:8px 10px;border-radius:10px;border:1px solid #ddd;font-size:13px;">
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

