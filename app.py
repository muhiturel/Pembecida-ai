import os, re, json
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Dict, List, Optional, Tuple

import requests
from lxml import etree
from difflib import SequenceMatcher

from fastapi import FastAPI
from fastapi.responses import Response, JSONResponse
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
    """
    Tüm yönlendirme linklerine UTM ekle.
    """
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
    t = el.text or ""
    return t.strip()


def first_text(node: etree._Element, tags: List[str], ns: Optional[Dict[str, str]] = None) -> str:
    """
    node altında tags listesindeki ilk dolu alanı döndür.
    tags: ["g:id", "id", "ID"] gibi kullanılabilir.
    """
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


# =========================
# Category / Intent logic
# =========================
# Kategori niyeti için konservatif sözlük (en kritikler)
# - Bunlar geçerse HARD FILTER uygulanır (yanlış ürün gelmesin diye)
CATEGORY_INTENTS = {
    # çanta ailesi
    "sirt_okul_cantasi": ["sırt çantası", "sirt cantasi", "okul çantası", "okul cantasi", "backpack", "school bag"],
    "cekcekli": ["çekçekli", "cekcekli", "çekçek", "cekcek", "trolley", "valizli", "kabin boy"],
    "beslenme": ["beslenme çantası", "beslenme cantasi", "lunch", "lunchbag"],

    # kırtasiye
    "kalem_kutusu": ["kalem kutusu", "kalemkutusu", "pencil case", "kalemlik"],
    "kalem": ["kalem", "pencil", "pen"],
    "silgi": ["silgi", "eraser"],
    "defter": ["defter", "günlük", "gunluk", "notebook", "diary"],
    "kirtasiye_set": ["kırtasiye set", "kirtasiye set", "setleri", "gift pack"],

    # içecek
    "suluk_termos": ["suluk", "termos", "matara", "bottle", "thermos"],

    # pop mart / oyuncak
    "blind_box": ["blind box", "sürpriz kutu", "surpriz kutu"],
    "oyuncak_figur": ["oyuncak", "figür", "figur", "koleksiyon", "collectible"],
    "labubu": ["labubu"],
    "crybaby": ["crybaby", "cry baby"],
    "skullpanda": ["skullpanda", "skull panda"],
    "hacipupu": ["hacipupu"],
    "twinkle": ["twinkle"],
    "dimoo": ["dimoo"],

    # aksesuar
    "kozmetik": ["kozmetik", "dudak", "parlatıcı", "parlatici", "lip", "gloss"],
    "anahtarlik": ["anahtarlık", "anahtarlik", "boyun askısı", "boyun askisi", "lanyard"],
    "cuzdan": ["cüzdan", "cuzdan", "wallet"],
    "pasaportluk": ["pasaportluk", "passport"],
    "taki": ["takı", "taki", "küpe", "kupe", "bileklik", "kolye", "jewelry", "bracelet", "earring"],
}

# Bu intent -> ürün category_path eşlemesi (kategori_path içinde geçmesi beklenen label’lar)
INTENT_TO_CATEGORY_PATH_HINTS = {
    "sirt_okul_cantasi": ["sırt & okul çantası", "okul-sirt-cantasi", "okul çantası", "sırt çantası", "canta"],
    "cekcekli": ["çekçekli çanta", "cekcekli", "trolley", "çocuk-çekçekli-valiz", "valiz"],
    "beslenme": ["beslenme çantası", "beslenme-cantasi"],
    "kalem_kutusu": ["kalem kutusu", "kalem-kutusu"],
    "kalem": ["kalem;"," kalem ", "kalem-"],
    "silgi": ["silgi"],
    "defter": ["defter", "günlük"],
    "kirtasiye_set": ["kırtasiye setleri", "kirtasiye-setleri"],
    "suluk_termos": ["suluk & termos", "suluk-termos", "termos", "suluk"],
    "blind_box": ["blind box", "sürpriz", "surpriz"],
    "oyuncak_figur": ["oyuncak figürler", "oyuncak ve figürler", "koleksiyonluk"],
    "labubu": ["labubu"],
    "crybaby": ["crybaby"],
    "skullpanda": ["skullpanda"],
    "hacipupu": ["hacipupu"],
    "twinkle": ["twinkle"],
    "dimoo": ["dimoo"],
    "kozmetik": ["kozmetik"],
    "anahtarlik": ["anahtarlık", "anahtarlik", "boyun askısı", "boyun askisi"],
    "cuzdan": ["cüzdan", "cuzdan"],
    "pasaportluk": ["pasaportluk"],
    "taki": ["takı", "taki", "küpe", "kupe", "bileklik"],
}

# Basit renk sözlüğü (type1/colors alanı için)
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
    # konservatif
    if "smiggle" in nq:
        return "smiggle"
    if "pop mart" in nq or "popmart" in nq:
        return "pop mart"
    return None


def conservative_brand_typo_fix(q: str) -> str:
    """
    Çok konservatif typo düzeltme: sadece marka benzeri kısa tokenlarda.
    Örn: smgle -> smiggle, cyrbaby -> crybaby (marka/seri gibi)
    """
    nq = norm(q)
    if not nq:
        return q

    tokens = nq.split()
    if len(tokens) > 4:
        return q

    known = ["smiggle", "crybaby", "labubu", "skullpanda", "hacipupu", "twinkle", "dimoo", "popmart", "pop", "mart"]
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
        # çok katı eşik
        if best_k and best_r >= 0.84 and t != best_k:
            new_tokens.append(best_k)
            changed = True
        else:
            new_tokens.append(t)

    if not changed:
        return q

    fixed = " ".join(new_tokens)
    # pop mart birleştir
    fixed = fixed.replace("pop mart", "pop mart").replace("popmart", "pop mart")
    return fixed


# =========================
# Feed indexing (robust)
# =========================
def fetch_and_index_feed() -> int:
    if not FEED_URL:
        raise RuntimeError("FEED_URL env is missing")

    r = requests.get(FEED_URL, timeout=120)
    r.raise_for_status()
    content = r.content

    xml = etree.fromstring(content)

    # Case A: Google Shopping style <item> + g: tags
    items = xml.findall(".//item")
    if items:
        ns = {"g": "http://base.google.com/ns/1.0"}
        products: List[Dict[str, Any]] = []

        for it in items:
            def gget(tag: str) -> str:
                v = first_text(it, [f"g:{tag}", tag], ns=ns)
                return v

            pid = gget("id")
            title = gget("title")
            brand = gget("brand")
            link = gget("link")
            price = gget("price")
            sale_price = gget("sale_price")
            product_type = gget("product_type")
            image = gget("image_link")
            add_imgs = all_texts(it, "g:additional_image_link", ns=ns) or all_texts(it, "additional_image_link")

            if not pid or not link:
                continue

            category_path = product_type or ""  # bazen "Aksesuar > Suluk & Termos" gibi gelir

            products.append({
                "id": str(pid).strip(),
                "code": safe_int(pid, 0),  # yoksa 0
                "brand": brand,
                "name": title,
                "title": title,
                "product_type": product_type,
                "category_path": category_path,
                "colors": [],
                "price": sale_price or price or "",
                "product_link": link,
                "link": add_utm(link),
                "image": image,
                "image_link": image,
                "additional_images": add_imgs,
                "additional_image_link": add_imgs,
                "search_text": " ".join([brand, title, product_type, link]).strip(),
            })

        # code büyük olan yeni (senin tanımına göre)
        products.sort(key=lambda x: x.get("code", 0), reverse=True)
        save_products(products)
        return len(products)

    # Case B: Custom feed - try to find product-like nodes
    # Heuristics: nodes named "product" or having child "product_link"/"link"
    candidates = xml.findall(".//product") + xml.findall(".//Product") + xml.findall(".//urun") + xml.findall(".//Urun")
    if not candidates:
        # fallback: anything that looks like it contains a link
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

        # colors: subproducts/type1 veya direct color alanları
        colors = []
        # subproducts/type1
        for sp in p.findall(".//subproducts") + p.findall(".//Subproducts") + p.findall(".//subproduct") + p.findall(".//Subproduct"):
            c = first_text(sp, ["type1", "Type1", "color", "Color", "renk", "Renk"])
            if c:
                colors.append(c)
        # direct
        c2 = first_text(p, ["type1", "Type1", "color", "Color", "renk", "Renk"])
        if c2:
            colors.append(c2)
        # uniq
        colors = list(dict.fromkeys([x.strip() for x in colors if x.strip()]))

        image = first_text(p, ["image", "Image", "image_link", "ImageLink", "img", "Img"])
        if not image:
            imgs = all_texts(p, "img") + all_texts(p, "image")
            image = imgs[0] if imgs else ""

        if not pid:
            # linkten id türetme (en son çare)
            pid = link or name
        if not link:
            continue

        products.append({
            "id": str(pid).strip(),
            "code": safe_int(code, safe_int(pid, 0)),
            "brand": brand,
            "name": name,
            "title": name,
            "category_path": category_path,
            "colors": colors,
            "price": price or "",
            "product_link": link,
            "link": add_utm(link),
            "image": image,
            "search_text": " ".join([brand, name, category_path, " ".join(colors)]).strip(),
        })

    # Sıralama: code büyük = yeni
    products.sort(key=lambda x: x.get("code", 0), reverse=True)
    save_products(products)
    return len(products)


# =========================
# Search
# =========================
def product_haystack(p: Dict[str, Any]) -> str:
    return norm(" ".join([
        p.get("brand", "") or "",
        p.get("name", "") or p.get("title", "") or "",
        p.get("category_path", "") or "",
        p.get("product_type", "") or "",
        " ".join(p.get("colors", []) or []),
        p.get("product_link", "") or p.get("link", "") or "",
    ]))


def search_products(products: List[Dict[str, Any]], q: str, k: int = 6) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Kategori + renk + marka niyetini yakalar.
    - Intent varsa: HARD FILTER (yanlış ürün gelmesin)
    - Intent yoksa: normal scoring
    """
    original_q = q or ""
    q2 = conservative_brand_typo_fix(original_q)
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

        # BRAND filter (konservatif): query içinde marka geçiyorsa diğer markaları ele
        if brand == "smiggle" and "smiggle" not in norm(p.get("brand", "")):
            continue
        if brand == "pop mart":
            b = norm(p.get("brand", ""))
            if ("pop" not in b) and ("mart" not in b) and ("pop mart" not in b):
                continue

        # INTENT hard filter
        if intents:
            cat = norm(p.get("category_path", "") or "")
            link = norm(p.get("product_link", "") or p.get("link", "") or "")
            name = norm(p.get("name", "") or p.get("title", "") or "")
            ok_any = False
            for ix in intents:
                hints = INTENT_TO_CATEGORY_PATH_HINTS.get(ix, [])
                # intent hintlerden biri category_path'te veya linkte geçsin, ya da isimde çok güçlü bir şekilde olsun
                if any(norm(h) in cat for h in hints) or any(norm(h) in link for h in hints):
                    ok_any = True
                    break
                # isim tabanlı güçlü eşleşme
                if ix == "suluk_termos" and (("termos" in name) or ("suluk" in name) or ("matara" in name)):
                    ok_any = True
                    break
                if ix == "kalem_kutusu" and (("kalem kutusu" in name) or ("kalemkutusu" in name)):
                    ok_any = True
                    break
                if ix == "cekcekli" and (("çekçek" in name) or ("cekcek" in name) or ("trolley" in name)):
                    ok_any = True
                    break
                if ix in ["labubu", "crybaby", "skullpanda", "hacipupu", "twinkle", "dimoo"] and (ix in name or ix in cat or ix in link):
                    ok_any = True
                    break

            if not ok_any:
                continue

        # COLOR filter/boost: renk istiyorsa renkler içinde yoksa ele (konservatif)
        if colors:
            pcols = " ".join([norm(c) for c in (p.get("colors") or [])])
            pname = norm(p.get("name", "") or p.get("title", "") or "")
            if not any((c in pcols) or (c in pname) for c in colors):
                # renk çok güçlü niyetse ele
                continue

        # Scoring
        score = 0

        # token match
        score += sum(1 for t in tokens if t in hay)

        # intent boost
        if intents:
            score += 5

        # color boost
        if colors:
            score += 3

        # recency boost: code büyük daha yeni
        code = safe_int(p.get("code", 0), 0)
        score += min(3, code // 200)  # kaba bir boost

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
    ok_feed = bool(FEED_URL)
    return {"ok": True, "feed": ok_feed}


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
    # sadece temel alanları döndür (debug okunaklı olsun)
    slim = []
    for p in hits:
        slim.append({
            "id": p.get("id"),
            "code": p.get("code"),
            "brand": p.get("brand"),
            "name": p.get("name") or p.get("title"),
            "colors": p.get("colors", []),
            "category_path": p.get("category_path", ""),
            "product_link": p.get("product_link") or p.get("link"),
        })
    return {"meta": meta, "hits": slim}


def to_widget_products(hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for p in hits:
        out.append({
            "title": p.get("name") or p.get("title") or "",
            "price": p.get("price") or "",
            "link": p.get("link") or add_utm(p.get("product_link") or ""),
            "image": p.get("image") or p.get("image_link") or "",
        })
    return out


@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()
    hits, meta = search_products(products, inp.query, k=6)

    # ✅ No hits message (senin istediğin metin + link mavi/altı çizgili widget CSS’te zaten var)
    if not hits:
        sss_url = "https://www.pembecida.com/sikca-sorulan-sorular"
        answer = (
            "Bu aramada eşleşen ürün bulamadım. "
            "İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). "
            f"İade/kargo gibi konular için <a href=\"{sss_url}\">Sıkça Sorulan Sorular</a> sayfamızı inceleyebilirsiniz."
        )
        return {"answer": answer, "products": [] , "meta": meta}

    # Widget’ta kartlar geleceği için LLM’ye sadece kısa bir “karşılama + yönlendirme” yazdırıyoruz.
    # (Ürün linklerini metin içine gömmüyoruz, yoksa üstteki “liste” geri gelir.)
    if client is None:
        # OpenAI yoksa fallback
        answer = "Sizin için birkaç uygun seçenek buldum. Aşağıdaki ürünleri inceleyebilirsiniz."
        return {"answer": answer, "products": to_widget_products(hits), "meta": meta}

    system = (
        "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
        "Kısa yanıt ver. Ürünleri metin içinde listeleme; sadece 1-2 cümleyle yönlendir ve aşağıdaki kartlara bakmasını söyle. "
        "Ürünler orijinaldir bilgisini ilk etkileşimlerde yumuşak şekilde hatırlatabilirsin; ama her cevapta tekrar etme. "
        "Fiyatlar değişmiş olabilir; en güncel bilgi ürün sayfasındadır. "
        "İade/kargo vb. sorularda SSS: https://www.pembecida.com/sikca-sorulan-sorular"
    )

    # meta’dan intent bilgisiyle daha doğru cümle
    intent_hint = ""
    if meta.get("intent"):
        intent_hint = f"Niyet: {', '.join(meta['intent'])}. "
    if meta.get("colors"):
        intent_hint += f"Renk: {', '.join(meta['colors'])}. "
    if meta.get("brand"):
        intent_hint += f"Marka: {meta['brand']}. "

    user = (
        f"Kullanıcı sorgusu: {inp.query}\n"
        f"Düzeltilmiş sorgu: {meta.get('query_fixed')}\n"
        f"{intent_hint}\n"
        "Kısa ve net bir yönlendirme cümlesi yaz. "
        "Metnin sonunda 'Aşağıdaki ürün kartlarından inceleyebilirsiniz.' gibi bir ifade olsun."
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
# Widget JS (API same-origin değil, render domain)
# =========================
@app.get("/pb-chat/widget.js")
def widget_js():
    # UTF-8 garanti
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

    return Response(js, media_type="application/javascript; charset=utf-8")
