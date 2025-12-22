import os, re, json
from difflib import SequenceMatcher
from urllib.parse import urlparse, urlencode, parse_qsl, urlunparse

import requests
from lxml import etree

from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI


# =========================
# App + CORS
# =========================
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =========================
# Config
# =========================
FEED_URL = os.environ.get("FEED_URL", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "products.json")

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


class ChatIn(BaseModel):
    query: str
    page_url: str | None = None


# =========================
# Helpers
# =========================
def norm(s: str) -> str:
    s = (s or "").lower()
    s = s.replace("ı", "i")  # aramada pratik
    s = re.sub(r"\s+", " ", s).strip()
    return s


def load_products():
    if not os.path.exists(STORE_PATH):
        return []
    try:
        with open(STORE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_products(products):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)


def add_utm(url: str) -> str:
    if not url:
        return url
    try:
        parts = urlparse(url)
        qs = dict(parse_qsl(parts.query))
        qs["utm_source"] = "pembegpt"
        qs["utm_medium"] = "chatbot"
        qs["utm_campaign"] = "pembecida"
        new_query = urlencode(qs, doseq=True)
        return urlunparse((parts.scheme, parts.netloc, parts.path, parts.params, new_query, parts.fragment))
    except Exception:
        return url


def localname(el) -> str:
    try:
        return etree.QName(el).localname.lower()
    except Exception:
        # fallback (tag "{ns}name" olabilir)
        t = str(getattr(el, "tag", "")).lower()
        if "}" in t:
            return t.split("}", 1)[1]
        return t


def safe_int(v, default=0):
    try:
        return int(str(v).strip())
    except Exception:
        return default


def clean_llm_answer(text: str) -> str:
    """
    LLM bazen ürünleri markdown liste olarak döküyor:
      [Ürün](https://www.pembecida.com/....) - 1234.00 TRY
    Kartları zaten ayrı bastığımız için bu satırları temizliyoruz.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.strip()

        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and ("[" in s and "](" in s):
            if ("try" in s.lower()) or ("tl" in s.lower()) or ("₺" in s) or ("price" in s.lower()):
                continue
            if s.startswith("-") or s.startswith("["):
                continue

        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and (s.startswith("-") or s.startswith("http")):
            continue

        out.append(ln)

    cleaned = "\n".join(out).strip()
    if len(cleaned) < 20:
        cleaned = (
            "Size uygun birkaç öneri çıkardım; aşağıda kartlar halinde görebilirsiniz. "
            "İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daraltayım."
        )
    return cleaned


# =========================
# Feed Parser (XML + JSON)
# =========================
def extract_from_json(data):
    """
    Şema bağımsız JSON parse:
    - products/product listelerini yakalamaya çalışır
    - her product için alanları normalize eder
    """
    # ürün listesine ulaş
    candidates = []
    if isinstance(data, list):
        candidates = data
    elif isinstance(data, dict):
        for k in ["products", "Products", "product", "Product", "items", "Items", "data", "Data"]:
            if k in data and isinstance(data[k], list):
                candidates = data[k]
                break
        if not candidates:
            # data.products gibi nested olabilir
            d = data.get("data") if isinstance(data.get("data"), dict) else None
            if d:
                for k in ["products", "product", "items"]:
                    if k in d and isinstance(d[k], list):
                        candidates = d[k]
                        break

    products = []
    for p in candidates or []:
        if not isinstance(p, dict):
            continue

        pid = str(p.get("id") or p.get("ID") or p.get("productId") or p.get("UrunID") or p.get("webserviskodu") or "").strip()
        title = str(p.get("title") or p.get("name") or p.get("UrunAdi") or p.get("Baslik") or "").strip()
        brand = str(p.get("brand") or p.get("Marka") or p.get("Brand") or "").strip()
        link = str(p.get("link") or p.get("url") or p.get("Url") or p.get("TamLink") or p.get("SeoLink") or "").strip()

        # fiyatlar
        price = str(p.get("price") or p.get("Price") or p.get("Fiyat") or p.get("SatisFiyati") or "").strip()
        sale_price = str(p.get("sale_price") or p.get("SalePrice") or p.get("IndirimliFiyat") or p.get("KampanyaFiyati") or "").strip()

        # kategori / tip
        product_type = str(p.get("product_type") or p.get("category") or p.get("Kategori") or p.get("CategoryPath") or "").strip()

        # görseller
        image = str(p.get("image_link") or p.get("image") or p.get("Image") or p.get("AnaResim") or "").strip()
        add_imgs = []
        imgs = p.get("additional_image_link") or p.get("images") or p.get("Images") or p.get("imgs")
        if isinstance(imgs, list):
            add_imgs = [str(x).strip() for x in imgs if str(x).strip()]

        # açıklama + meta
        description = str(p.get("description") or p.get("Description") or p.get("UrunAciklama") or "").strip()
        meta_title = str(p.get("meta_title") or p.get("MetaTitle") or p.get("SeoTitle") or "").strip()
        meta_desc = str(p.get("meta_description") or p.get("MetaDescription") or p.get("SeoDescription") or "").strip()
        meta_keys = str(p.get("meta_keywords") or p.get("MetaKeywords") or p.get("SeoKeywords") or "").strip()

        # code (sıra numarası)
        code = safe_int(p.get("code") or p.get("Code") or p.get("sira") or p.get("Sira") or 0, 0)

        # subproducts + renk (type1)
        colors = []
        subp = p.get("subproducts") or p.get("SubProducts") or p.get("subProducts")
        if isinstance(subp, list):
            for sp in subp:
                if isinstance(sp, dict):
                    c = str(sp.get("type1") or sp.get("Type1") or sp.get("renk") or sp.get("Renk") or "").strip()
                    if c:
                        colors.append(c)

        # minimum
        if not pid or not link:
            continue

        products.append({
            "id": pid,
            "code": code,
            "title": title,
            "brand": brand,
            "brand_title": f"{brand} {title}".strip(),
            "link": link,
            "price": price,
            "sale_price": sale_price,
            "product_type": product_type,
            "description": description,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "meta_keywords": meta_keys,
            "color_terms": list(dict.fromkeys([c for c in colors if c])),
            "image_link": image,
            "additional_image_link": add_imgs,
        })

    return products


def extract_from_xml(xml_root):
    """
    Şema bağımsız XML parse:
    - tüm node’ları dolaşır
    - localname’i product/urun/item benzeri olan node’ları product kabul eder
    - alt node/attribute’lardan alanları çıkarır
    """
    def find_text(node, possible_names):
        # child tag matches (namespace bağımsız)
        for child in node.iter():
            ln = localname(child)
            if ln in possible_names:
                tx = (child.text or "").strip()
                if tx:
                    return tx
        # attribute matches
        for k, v in (node.attrib or {}).items():
            if k.lower() in possible_names and str(v).strip():
                return str(v).strip()
        return ""

    def find_many_texts(node, possible_names):
        out = []
        for child in node.iter():
            if localname(child) in possible_names:
                tx = (child.text or "").strip()
                if tx:
                    out.append(tx)
        # unique preserve
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    # product node adayları
    product_nodes = []
    for el in xml_root.iter():
        ln = localname(el)
        if ln in ("product", "urun", "item"):
            product_nodes.append(el)

    # bazen <products><productlist><p> gibi garip olabilir: çok node gelirse yanlış yakalamayı azalt
    if len(product_nodes) > 20000:
        # daha sıkı filtre: içinde link/url benzeri bir alan varsa product say
        filtered = []
        for el in product_nodes:
            txt = etree.tostring(el, encoding="unicode", with_tail=False)
            if ("http" in txt) and (("link" in txt.lower()) or ("url" in txt.lower()) or ("tamlink" in txt.lower()) or ("seolink" in txt.lower())):
                filtered.append(el)
        product_nodes = filtered

    products = []
    for p in product_nodes:
        pid = find_text(p, ["id", "urunid", "productid", "webserviskodu", "stok", "stokkodu", "modelkodu"])
        title = find_text(p, ["title", "name", "urunadi", "baslik", "ad"])
        brand = find_text(p, ["brand", "marka"])
        link = find_text(p, ["link", "url", "tamlink", "seolink", "dislink"])

        price = find_text(p, ["price", "fiyat", "satisfiyati", "satis", "satis_fiyat", "saleprice", "sale_price"])
        sale_price = find_text(p, ["sale_price", "saleprice", "indirimlifiyat", "kampanyafiyati"])

        product_type = find_text(p, ["product_type", "kategori", "category", "categorypath", "kategoriyolu"])

        image = find_text(p, ["image_link", "image", "resim", "anaresim", "mainimage"])
        add_imgs = find_many_texts(p, ["additional_image_link", "image", "resim", "img", "images"])

        description = find_text(p, ["description", "aciklama", "urunaciklama", "detay"])
        meta_title = find_text(p, ["meta_title", "seotitle", "titlemeta"])
        meta_desc = find_text(p, ["meta_description", "seodescription", "descriptionmeta", "metadescription"])
        meta_keys = find_text(p, ["meta_keywords", "seokeywords", "keywords", "metakeywords"])

        code = safe_int(find_text(p, ["code", "sirano", "sira", "sort", "order"]), 0)

        # subproducts/type1 renk
        colors = []
        # subproducts node’u varsa ara
        for sp in p.iter():
            if localname(sp) in ("subproducts", "subproduct", "alturun", "alturunler"):
                # type1 çocuklarında
                for ch in sp.iter():
                    if localname(ch) in ("type1", "renk", "color"):
                        tx = (ch.text or "").strip()
                        if tx:
                            colors.append(tx)

        if not pid or not link:
            continue

        products.append({
            "id": pid,
            "code": code,
            "title": title,
            "brand": brand,
            "brand_title": f"{brand} {title}".strip(),
            "link": link,
            "price": price,
            "sale_price": sale_price,
            "product_type": product_type,
            "description": description,
            "meta_title": meta_title,
            "meta_description": meta_desc,
            "meta_keywords": meta_keys,
            "color_terms": list(dict.fromkeys([c for c in colors if c])),
            "image_link": image,
            "additional_image_link": add_imgs,
        })

    return products


def fetch_and_index_feed() -> int:
    if not FEED_URL:
        raise RuntimeError("FEED_URL env is missing")

    r = requests.get(FEED_URL, timeout=180)
    r.raise_for_status()

    raw = r.content
    head = raw.lstrip()[:1]

    products = []

    # JSON olabilir
    if head in (b"{", b"["):
        try:
            data = r.json()
            products = extract_from_json(data)
        except Exception:
            products = []
    else:
        # XML
        try:
            xml = etree.fromstring(raw)
            products = extract_from_xml(xml)
        except Exception:
            products = []

    save_products(products)
    return len(products)


def ensure_products_loaded():
    products = load_products()
    if products:
        return products
    try:
        fetch_and_index_feed()
    except Exception:
        pass
    return load_products()


# =========================
# Typo Fixing
# =========================
def fix_brand_typo(q: str) -> str:
    nq = norm(q)
    if not nq:
        return q
    tokens = nq.split()
    if len(tokens) > 3:
        return q

    known_single = ["smiggle", "popmart", "pembecida", "pop", "mart"]
    new_tokens = []
    changed = False

    for t in tokens:
        best_ratio, best_k = 0.0, None
        for k in known_single:
            r = SequenceMatcher(None, t, k).ratio()
            if r > best_ratio:
                best_ratio, best_k = r, k
        if best_k and len(t) >= 4 and best_ratio >= 0.82:
            new_tokens.append(best_k)
            changed = True
        else:
            new_tokens.append(t)

    # "pop mart" birleşik yazım
    joined = " ".join(new_tokens)
    joined = joined.replace("pop mart", "popmart")
    return joined if changed else q


def fix_keyword_typo(q: str) -> str:
    nq = norm(q)
    if not nq:
        return q

    tokens = nq.split()
    if len(tokens) > 4:
        return q

    known_kw_single = [
        "crybaby", "labubu", "skullpanda", "hirono", "dimoo", "molly",
        "kalem", "kalemkutusu", "canta", "çanta", "sirt", "sırt", "cekcek", "çekçek"
    ]

    new_tokens = []
    changed = False

    for t in tokens:
        if len(t) < 4:
            new_tokens.append(t)
            continue

        best_ratio, best_k = 0.0, None
        for k in known_kw_single:
            r = SequenceMatcher(None, t, k).ratio()
            if r > best_ratio:
                best_ratio, best_k = r, k

        if best_k and best_ratio >= 0.84:
            new_tokens.append(best_k)
            changed = True
        else:
            new_tokens.append(t)

    return " ".join(new_tokens) if changed else q


# =========================
# Search
# =========================
def simple_search(products, q, k=6):
    nq = norm(q)
    if not nq:
        return []

    tokens = nq.split()

    scored = []
    for p in products:
        hay = norm(" ".join([
            p.get("title", ""),
            p.get("brand_title", ""),
            p.get("product_type", ""),
            p.get("description", ""),
            p.get("meta_title", ""),
            p.get("meta_description", ""),
            p.get("meta_keywords", ""),
            " ".join(p.get("color_terms", []) or []),
            p.get("link", ""),
        ]))

        score = 0
        score += sum(1 for t in tokens if t in hay)

        # renk niyetini güçlendir (pembe çanta vs)
        if any(t in ["pembe", "mavi", "siyah", "beyaz", "kirmizi", "kırmızı", "yesil", "yeşil", "lila", "gri"] for t in tokens):
            score += 2 * sum(1 for t in tokens if t in norm(" ".join(p.get("color_terms", []) or [])))

        if score > 0:
            # code yeni ürün: hafif bonus (yüksek code daha yeni)
            score += min(3, safe_int(p.get("code", 0)) // 100000)
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def fuzzy_typo_suggest(products, q: str, k: int = 3):
    """
    Çok konservatif typo toleransı: emin değilse önermesin.
    """
    nq = norm(q)
    if not nq or len(nq) < 3:
        return []

    scored = []
    for p in products:
        hay = norm(" ".join([
            p.get("title", ""),
            p.get("brand_title", ""),
            p.get("product_type", ""),
            " ".join(p.get("color_terms", []) or []),
        ]))
        if not hay:
            continue
        r = SequenceMatcher(None, nq, hay).ratio()
        scored.append((r, p))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)
    best_score, best_p = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    if best_score >= 0.88 and (best_score - second_score) >= 0.04:
        out = [best_p]
        for s, p in scored[1:]:
            if len(out) >= k:
                break
            if s >= 0.86:
                out.append(p)
        return out

    return []


# =========================
# API Endpoints
# =========================
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


@app.get("/pb-chat/debug/sample")
def debug_sample(limit: int = 10):
    products = ensure_products_loaded()
    return {"count": len(products), "sample": products[:limit]}


@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = ensure_products_loaded()
    rows = []
    for p in products[:limit]:
        rows.append({
            "id": p.get("id"),
            "code": p.get("code"),
            "brand": p.get("brand"),
            "title": p.get("title"),
            "colors": p.get("color_terms", []),
            "product_type": p.get("product_type"),
            "link": p.get("link"),
            "image_link": p.get("image_link"),
        })
    return {"count": len(products), "rows": rows}


@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = ensure_products_loaded()
    if not products:
        return {
            "answer": "Şu an ürün listesini yükleyemedim. Lütfen biraz sonra tekrar dener misiniz?",
            "products": []
        }

    q_fixed = fix_brand_typo(inp.query)
    q_fixed = fix_keyword_typo(q_fixed)

    hits = simple_search(products, q_fixed, k=6)

    if not hits:
        typo_hits = fuzzy_typo_suggest(products, inp.query, k=3)
        if typo_hits:
            hits = typo_hits
        else:
            return {
                "answer": (
                    'Bu aramada eşleşen ürün bulamadım. '
                    'İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). '
                    f'İade/kargo gibi konular için <a href="{add_utm("https://www.pembecida.com/sikca-sorulan-sorular")}">Sıkça Sorulan Sorular</a> sayfamızı inceleyebilirsiniz.'
                ),
                "products": []
            }

    top = hits[:5]

    ui_products = []
    for p in top:
        url = add_utm(p.get("link") or "")
        price = (p.get("sale_price") or p.get("price") or "")
        ui_products.append({
            "title": p.get("brand_title") or p.get("title") or "",
            "link": url,
            "price": price,
            "image": p.get("image_link") or "",
        })

    # OpenAI yoksa da çalışsın
    if not client:
        return {
            "answer": "Size uygun birkaç öneri çıkardım; aşağıda kartlar halinde görebilirsiniz. İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daraltayım.",
            "products": ui_products
        }

    safe_products = [
        {
            "title": (p.get("brand_title") or "").strip(),
            "price": ((p.get("sale_price") or p.get("price") or "")).strip(),
            "link": add_utm((p.get("link") or "").strip()),
            "product_type": (p.get("product_type") or "").strip(),
            "colors": (p.get("color_terms") or []),
            "code": p.get("code", 0),
        }
        for p in top
    ]

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
        "products": safe_products,
        "note": "Ürünleri asla metinle listeleme; sadece kısa karşılama yaz. Kartlar ayrıca gösterilecek."
    }

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ],
    )

    answer = clean_llm_answer(resp.output_text)
    return {"answer": answer, "products": ui_products}


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
      z-index:99999;
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
  box.style.cssText = "display:none;position:fixed;right:16px;bottom:64px;z-index:99999;width:340px;max-width:calc(100vw - 32px);height:520px;background:#fff;border:2px solid #9EA3A8;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;font-family:system-ui;";
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
    return Response(js, media_type="application/javascript")
