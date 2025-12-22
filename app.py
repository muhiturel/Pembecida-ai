import os
import re
import json
import html
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse

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
# Helpers
# -----------------------------
class ChatIn(BaseModel):
    query: str
    page_url: str | None = None


def norm(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_html(s: str) -> str:
    s = s or ""
    s = re.sub(r"<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def safe_text(el: Optional[etree._Element]) -> str:
    if el is None:
        return ""
    txt = el.text or ""
    return txt.strip()


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


def format_price(p: Dict[str, Any]) -> str:
    # Yeni XML’de vat_included alanları var: price_special_vat_included / price_list_vat_included
    # yoksa price_special / price_list kullan
    cur = (p.get("currency") or "TL").strip()
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
                # TL için 2 basamak
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
        <code>...</code>
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
            ...
          </subproduct>
        </subproducts>
      </product>
    """
    root = etree.fromstring(xml_bytes)

    products_out: List[Dict[str, Any]] = []

    # Bazı XML’lerde root farklı olabilir; yine de tüm product’ları gez
    for prod in root.findall(".//product"):
        code = safe_text(prod.find("code"))
        name = safe_text(prod.find("name"))
        brand = safe_text(prod.find("brand"))
        link = safe_text(prod.find("product_link"))
        category_path = safe_text(prod.find("category_path"))
        model = safe_text(prod.find("model"))

        # fiyatlar
        price_list = safe_text(prod.find("price_list"))
        price_list_vat_included = safe_text(prod.find("price_list_vat_included"))
        price_special = safe_text(prod.find("price_special"))
        price_special_vat_included = safe_text(prod.find("price_special_vat_included"))
        currency = safe_text(prod.find("currency"))
        stock = safe_text(prod.find("stock"))

        detail_html = safe_text(prod.find("detail"))
        seo_title = safe_text(prod.find("seo_title"))
        seo_description = safe_text(prod.find("seo_description"))

        # images
        images = []
        for img in prod.findall(".//images/img_item"):
            t = safe_text(img)
            if t:
                images.append(t)

        # subproducts / colors(type1)
        colors = []
        subproducts = []
        for sp in prod.findall(".//subproducts/subproduct"):
            sp_code = safe_text(sp.find("code"))
            sp_ws_code = safe_text(sp.find("ws_code"))
            sp_type1 = safe_text(sp.find("type1"))  # renk
            sp_type2 = safe_text(sp.find("type2"))
            sp_barcode = safe_text(sp.find("barcode"))
            sp_stock = safe_text(sp.find("stock"))
            sp_price_list = safe_text(sp.find("price_list"))
            sp_price_special = safe_text(sp.find("price_special"))

            if sp_type1:
                colors.append(sp_type1)

            subproducts.append(
                {
                    "code": sp_code,
                    "ws_code": sp_ws_code,
                    "type1": sp_type1,
                    "type2": sp_type2,
                    "barcode": sp_barcode,
                    "stock": sp_stock,
                    "price_list": sp_price_list,
                    "price_special": sp_price_special,
                }
            )

        # minimum alanlar
        if not link or not name:
            continue

        # index text
        detail_text = strip_html(detail_html)
        seo_text = strip_html(seo_title + " " + seo_description)

        # Linke UTM ekle (index’te ve widget kartlarında tek kaynak olsun)
        link_utm = add_utm(link)

        products_out.append(
            {
                "code": code,  # senin dediğin: büyükse daha yeni
                "name": name,
                "brand": brand,
                "brand_name": f"{brand} {name}".strip(),
                "product_link": link_utm,
                "category_path": category_path,
                "model": model,
                "stock": stock,
                "currency": currency or "TL",
                "price_list": price_list,
                "price_list_vat_included": price_list_vat_included,
                "price_special": price_special,
                "price_special_vat_included": price_special_vat_included,
                "images": images,
                "colors": list(dict.fromkeys([c.strip() for c in colors if c.strip()])),
                "subproducts": subproducts,
                "detail": detail_html,
                "seo_title": seo_title,
                "seo_description": seo_description,
                "search_text": norm(
                    " ".join(
                        [
                            name,
                            brand,
                            category_path,
                            model,
                            " ".join(colors),
                            detail_text,
                            seo_text,
                            link,
                        ]
                    )
                ),
            }
        )

    # code büyük olan daha yeni: büyükten küçüğe sırala
    def code_int(x: Dict[str, Any]) -> int:
        try:
            return int(str(x.get("code") or "0").strip())
        except Exception:
            return 0

    products_out.sort(key=code_int, reverse=True)
    return products_out


def fetch_and_index_feed() -> Tuple[bool, int, str]:
    if not FEED_URL:
        save_products([])
        return False, 0, "FEED_URL boş"

    r = requests.get(FEED_URL, timeout=120)
    r.raise_for_status()

    products = parse_products_from_xml(r.content)
    save_products(products)
    return True, len(products), "OK"


# -----------------------------
# Search (token + color + typo)
# -----------------------------
COLOR_HINTS = {
    "pembe": ["pembe", "pink"],
    "mor": ["mor", "purple"],
    "mavi": ["mavi", "blue"],
    "lacivert": ["lacivert", "navy"],
    "siyah": ["siyah", "black"],
    "beyaz": ["beyaz", "white"],
    "kırmızı": ["kırmızı", "kirmizi", "red"],
    "yeşil": ["yeşil", "yesil", "green"],
    "lila": ["lila", "lilac"],
    "mint": ["mint"],
    "gold": ["gold", "altın", "altin"],
}

INTENT_MAP = {
    "kalem_kutusu": ["kalem kutusu", "kalemkutusu", "pencil case", "case"],
    "sirt_cantasi": ["sırt çantası", "sirt cantasi", "backpack", "okul çantası", "okul cantasi"],
    "cekcekli": ["çekçekli", "cekcekli", "trolley"],
    "beslenme": ["beslenme çantası", "beslenme cantasi", "lunch bag", "lunchbox"],
    "suluk": ["suluk", "matara", "bottle"],
}


def build_vocab(products: List[Dict[str, Any]]) -> List[str]:
    vocab = set()
    for p in products:
        for w in norm(p.get("brand", "")).split():
            if len(w) >= 4:
                vocab.add(w)
        for w in norm(p.get("name", "")).split():
            if len(w) >= 4:
                vocab.add(w)
        for c in p.get("colors", []) or []:
            cw = norm(c)
            if len(cw) >= 3:
                vocab.add(cw)
    return sorted(vocab)


def correct_tokens(tokens: List[str], vocab: List[str]) -> Tuple[List[str], List[Tuple[str, str]]]:
    """
    Basit typo düzeltme:
    - sadece 4+ harfli tokenlarda dener
    - difflib ile en yakın eşleşmeyi alır
    - agresif davranmasın diye cutoff yüksek
    """
    import difflib

    corrected = []
    changes = []
    for t in tokens:
        if len(t) < 4:
            corrected.append(t)
            continue
        m = difflib.get_close_matches(t, vocab, n=1, cutoff=0.86)
        if m and m[0] != t:
            corrected.append(m[0])
            changes.append((t, m[0]))
        else:
            corrected.append(t)
    return corrected, changes


def simple_search(products: List[Dict[str, Any]], q: str, k: int = 8) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    nq = norm(q)
    if not nq:
        return [], {"used_query": "", "typo": []}

    tokens = nq.split()
    vocab = build_vocab(products)
    tokens2, typo_changes = correct_tokens(tokens, vocab)

    # intent terimleri
    intent_terms = []
    for terms in INTENT_MAP.values():
        if any(norm(t) in nq for t in terms):
            intent_terms += [norm(t) for t in terms]

    # color terms
    color_terms = []
    for _, terms in COLOR_HINTS.items():
        if any(norm(t) in nq for t in terms):
            color_terms += [norm(t) for t in terms]

    scored: List[Tuple[int, Dict[str, Any]]] = []
    for p in products:
        hay = p.get("search_text", "")

        score = 0
        # token match
        score += sum(2 for t in tokens2 if t and t in hay)

        # intent boost
        score += sum(3 for t in intent_terms if t and t in hay)

        # color boost: renk hem hay içinde hem colors içinde ise daha güçlü
        p_colors = norm(" ".join(p.get("colors", []) or []))
        score += sum(4 for t in color_terms if t and (t in p_colors or t in hay))

        # brand boost (Smiggle / Pop Mart vb.)
        b = norm(p.get("brand", ""))
        score += 2 if b and b in nq else 0

        # ürün kodu / ws_code / barcode araması (kullanıcı bazen kod yazar)
        if any(x in nq for x in [norm(str(p.get("code") or "")), norm(str(p.get("product_link") or ""))]):
            score += 1

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [p for _, p in scored[:k]]

    meta = {
        "used_query": " ".join(tokens2),
        "typo": typo_changes,  # [('cyrbaby','crybaby')] gibi
    }
    return hits, meta


# -----------------------------
# API routes
# -----------------------------
@app.get("/pb-chat/health")
def health():
    products = load_products()
    return {"ok": True, "feed_configured": bool(FEED_URL), "count": len(products)}


@app.get("/pb-chat/reindex")
def reindex_get():
    ok, count, msg = fetch_and_index_feed()
    return {"ok": ok, "count": count, "message": msg}


@app.post("/pb-chat/reindex")
def reindex_post():
    ok, count, msg = fetch_and_index_feed()
    return {"ok": ok, "count": count, "message": msg}


@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = load_products()
    rows = []
    for p in products[: max(0, limit)]:
        rows.append(
            {
                "code": p.get("code"),
                "brand": p.get("brand"),
                "name": p.get("name"),
                "colors": p.get("colors"),
                "category_path": p.get("category_path"),
                "product_link": p.get("product_link"),
                "img0": (p.get("images") or [None])[0],
            }
        )
    return {"count": len(products), "rows": rows}


@app.get("/pb-chat/debug/search")
def debug_search(q: str, k: int = 10):
    products = load_products()
    hits, meta = simple_search(products, q, k=k)
    # hafifletilmiş çıktı
    lite = []
    for p in hits:
        lite.append(
            {
                "code": p.get("code"),
                "brand": p.get("brand"),
                "name": p.get("name"),
                "colors": p.get("colors"),
                "product_link": p.get("product_link"),
            }
        )
    return {"query": q, "meta": meta, "hits": lite}


def no_match_message() -> str:
    # Kullanıcının istediği metin + HTML link
    return (
        'Bu aramada eşleşen ürün bulamadım. '
        'İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). '
        'İade/kargo gibi konular için '
        '<a href="https://www.pembecida.com/sikca-sorulan-sorular" target="_self">Sıkça Sorulan Sorular</a> '
        'sayfamızı inceleyebilirsiniz.'
    )


def llm_answer(user_query: str, typo_meta: Dict[str, Any]) -> str:
    # LLM yoksa sabit, kısa ve güvenli bir metin dön
    typo_note = ""
    if typo_meta.get("typo"):
        # emin olmadığımızda gereksiz yönlendirme yapmayalım; sadece “bunu mu demek istediniz?” tonu
        pairs = typo_meta["typo"][:1]
        typo_note = f' (Bu aramada “{pairs[0][0]}” yazdınız; “{pairs[0][1]}” demek istemiş olabilirsiniz.)'

    if not client:
        return (
            "Merhaba! Size yardımcı olayım. Ne arıyorsunuz?"
            " Örn: “Smiggle kalem kutusu”, “8 yaş hediye”, “Pop Mart blind box”."
            + typo_note
        )

    system = (
        "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
        "Yanıtlar kısa ve net olsun. Kullanıcıdan arama niyetini netleştirmek için gerekiyorsa 1 soru sor. "
        "Ürün linkleri verilecek; linklere ek yorum yaparken fiyatların değişebileceğini nazikçe belirt. "
        "İade/kargo vb. sorularda SSS sayfasına yönlendir: https://www.pembecida.com/sikca-sorulan-sorular ."
    )

    user = (
        f"Kullanıcı sorusu: {user_query}\n"
        f"Not: Typo ihtimali: {typo_meta.get('typo', [])}\n\n"
        "Cevabın sadece metin olsun; ürün listeleme yapma. "
        "Kullanıcı ürün arıyorsa tek cümlelik karşılama + arama örneği ver."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.output_text.strip()


@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()
    hits, meta = simple_search(products, inp.query, k=6)

    if not hits:
        # typo meta varsa, yine de emin değilsek bulamadım demek istiyorsun → burada kesin bulamadıysa direkt mesaj
        return {"answer": no_match_message(), "products": [], "meta": meta}

    # Widget kartları için sade ürün objeleri
    cards = []
    for p in hits[:5]:
        cards.append(
            {
                "title": p.get("brand_name") or p.get("name") or "",
                "link": p.get("product_link") or "",
                "price": format_price(p),
                "image": (p.get("images") or [None])[0],
                "code": p.get("code"),
                "colors": p.get("colors") or [],
            }
        )

    answer = llm_answer(inp.query, meta)
    return {"answer": answer, "products": cards, "meta": meta}


@app.get("/pb-chat/widget.js")
def widget():
    # Not: widget.js zaten sende ayrı güncellendi; burada sadece servis etmeye devam ediyoruz.
    # Eğer widget.js'i repo içinde ayrı dosyaya taşırsan burayı güncelleyebiliriz.
    js = """
// widget.js endpointi bu repoda özelleştirildiği için burada minimal bırakıldı.
// Render tarafında mevcut widget.js kodunu kullanmaya devam ediyorsun.
console.log("pb-chat/widget.js loaded");
""".strip()
    return Response(js, media_type="application/javascript")
