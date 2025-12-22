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

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Config ----------
FEED_URL = os.environ.get("FEED_URL", "").strip()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "products.json")

OPENAI_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
client = OpenAI(api_key=OPENAI_KEY) if OPENAI_KEY else None


class ChatIn(BaseModel):
    query: str
    page_url: str | None = None


def norm(s: str) -> str:
    s = (s or "").lower()
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
    """UTM garantisi (backend)."""
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


def fetch_and_index_feed() -> int:
    if not FEED_URL:
        raise RuntimeError("FEED_URL env is missing")

    r = requests.get(FEED_URL, timeout=120)
    r.raise_for_status()

    xml = etree.fromstring(r.content)
    ns = {"g": "http://base.google.com/ns/1.0"}

    # 1) Önce en yaygın düğümleri dene
    items = xml.findall(".//item")  # Google Shopping / RSS benzeri
    if not items:
        # T-Soft / custom XML varyasyonları
        candidates = [".//Urun", ".//urun", ".//Product", ".//product", ".//products/*", ".//Items/*"]
        for path in candidates:
            items = xml.findall(path)
            if items:
                break

    def find_text(node, tag_names):
        """Birden fazla olası tag adından ilk dolu olanı bulur (namespace'li ve namespacesiz)."""
        for t in tag_names:
            # google namespace dene
            el = node.find(f"g:{t}", namespaces=ns)
            if el is not None and (el.text or "").strip():
                return (el.text or "").strip()

            # namespacesiz
            el2 = node.find(t)
            if el2 is not None and (el2.text or "").strip():
                return (el2.text or "").strip()

            # bazı XML’lerde tag’ler nested/uppercase olabilir, geniş arama:
            el3 = node.find(f".//{t}")
            if el3 is not None and (el3.text or "").strip():
                return (el3.text or "").strip()

        return ""

    def find_all_texts(node, tag_names):
        out = []
        for t in tag_names:
            # namespace'li
            for el in node.findall(f"g:{t}", namespaces=ns):
                tx = (el.text or "").strip()
                if tx:
                    out.append(tx)
            # namespacesiz
            for el in node.findall(t):
                tx = (el.text or "").strip()
                if tx:
                    out.append(tx)
            # geniş
            for el in node.findall(f".//{t}"):
                tx = (el.text or "").strip()
                if tx:
                    out.append(tx)
        # unique koru, sırayı bozma
        seen = set()
        uniq = []
        for x in out:
            if x not in seen:
                uniq.append(x)
                seen.add(x)
        return uniq

    products = []
    for it in items:
        pid = find_text(it, ["id", "ID", "UrunID", "urun_id", "ProductId", "product_id", "StokKodu", "ModelKodu", "webserviskodu"])
        title = find_text(it, ["title", "Title", "Baslik", "Ad", "UrunAdi", "name"])
        brand = find_text(it, ["brand", "Brand", "Marka", "brand_name"])
        link = find_text(it, ["link", "Link", "Url", "url", "SeoLink", "TamLink", "Dislink", "ProductUrl"])
        price = find_text(it, ["price", "Price", "Fiyat", "SatisFiyati", "satisfiyati", "sale_price", "SalePrice"])
        sale_price = find_text(it, ["sale_price", "SalePrice", "IndirimliFiyat", "KampanyaFiyati"])
        product_type = find_text(it, ["product_type", "ProductType", "Kategori", "Category", "category_path", "KategoriYolu"])
        image = find_text(it, ["image_link", "image", "Image", "Resim", "MainImage", "AnaResim", "image_url"])

        add_imgs = find_all_texts(it, ["additional_image_link", "AdditionalImage", "Resim", "Images", "image", "Image"])

        # minimum alanlar
        if not pid:
            # bazı feedlerde id attribute olarak gelir: <Urun id="123">
            pid = (it.get("id") or it.get("ID") or "").strip()

        if not pid or not link:
            continue

        products.append({
            "id": pid,
            "title": title,
            "brand": brand,
            "brand_title": f"{brand} {title}".strip(),
            "link": link,
            "price": price,
            "sale_price": sale_price,
            "product_type": product_type,
            "image_link": image,
            "additional_image_link": add_imgs,
        })

    save_products(products)
    return len(products)



def ensure_products_loaded():
    """Ürünler boşsa otomatik reindex (deploy sonrası products.json sıfırlanabiliyor)."""
    products = load_products()
    if products:
        return products

    try:
        fetch_and_index_feed()
    except Exception:
        # FEED erişimi vs hata varsa sessiz geç; chat tarafında fallback döner
        pass
    return load_products()


# ---------- Search ----------
def simple_search(products, q, k=6):
    nq = norm(q)
    if not nq:
        return []

    intent_map = {
        "kalem_kutusu": ["kalem", "kalemkutusu", "kalem kutusu", "kalemlik", "pencil", "case", "pencil case"],
        "sirt_cantasi": ["sırt", "sirt", "backpack", "çanta", "canta"],
        "cekcekli": ["çekçek", "cekcek", "trolley", "çekçekli", "cekcekli"],
        "beslenme": ["beslenme", "lunch", "lunchbox", "food", "beslenme çantası", "beslenme cantasi"],
        "suluk": ["suluk", "bottle", "matara"],
    }

    active_terms = []
    for _, terms in intent_map.items():
        if any(t in nq for t in terms):
            active_terms += terms

    tokens = nq.split()
    scored = []

    for p in products:
        hay = norm(" ".join([
            p.get("title", ""),
            p.get("brand_title", ""),
            p.get("product_type", ""),
            p.get("link", ""),
        ]))

        score = 0
        score += sum(1 for t in tokens if t in hay)
        if active_terms:
            score += 3 * sum(1 for t in active_terms if t in hay)

        if score > 0:
            scored.append((score, p))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]


def fix_brand_typo(q: str) -> str:
    """
    Kısa typo'larda marka düzeltme (konservatif).
    smgle/smigle -> smiggle gibi.
    """
    nq = norm(q)
    if not nq:
        return q

    tokens = nq.split()
    if len(tokens) > 3:
        return q

    known_single = ["smiggle", "popmart", "pembecida"]
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

    return " ".join(new_tokens) if changed else q


def fix_keyword_typo(q: str) -> str:
    """
    Seri/keyword typo düzeltme (konservatif).
    cyrbaby -> crybaby gibi.
    """
    nq = norm(q)
    if not nq:
        return q

    tokens = nq.split()
    if len(tokens) > 4:
        return q

    known_kw_single = [
        "crybaby", "labubu", "skullpanda", "hirono", "dimoo", "molly",
        "pencil", "backpack"
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

        # cyrbaby/crybby -> crybaby için yeterince katı eşik
        if best_k and best_ratio >= 0.84:
            new_tokens.append(best_k)
            changed = True
        else:
            new_tokens.append(t)

    return " ".join(new_tokens) if changed else q


def fuzzy_typo_suggest(products, q: str, k: int = 3):
    """
    Genel typo toleransı: ÇOK konservatif.
    Şüphe varsa hiç önermeyip 'bulamadım' der.
    """
def clean_llm_answer(text: str) -> str:
    """
    LLM bazen ürünleri markdown liste olarak döküyor:
      [Ürün](https://www.pembecida.com/....) - 1234.00 TRY
    Biz kartları zaten ayrı bastığımız için bu satırları temizliyoruz.
    """
    if not text:
        return ""

    lines = text.splitlines()
    out = []
    for ln in lines:
        s = ln.strip()

        # Markdown link + pembecida URL + fiyat/Try içeriyorsa -> at
        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and ("[" in s and "](" in s):
            # çoğu zaman "- 1234.00 TRY" gibi biter
            if ("try" in s.lower()) or ("tl" in s.lower()) or ("₺" in s) or ("price" in s.lower()):
                continue
            # fiyat olmasa bile ürün linki liste gibi geldiyse yine at
            if s.startswith("-") or s.startswith("["):
                continue

        # Düz URL basmışsa da (liste gibi) temizleyelim
        if ("https://www.pembecida.com/" in s or "https://pembecida.com/" in s) and (s.startswith("-") or s.startswith("http")):
            continue

        out.append(ln)

    cleaned = "\n".join(out).strip()

    # Çok kısa kaldıysa güvenli bir karşılama koy
    if len(cleaned) < 20:
        cleaned = "Size uygun birkaç öneri çıkardım; aşağıda kartlar halinde görebilirsiniz. İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daraltayım."

    return cleaned
    
    
    nq = norm(q)
    if not nq or len(nq) < 3:
        return []

    scored = []
    for p in products:
        hay = norm(" ".join([
            p.get("title", ""),
            p.get("brand_title", ""),
            p.get("product_type", ""),
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

    # Çok katı eşik: güven yoksa önerme
    if best_score >= 0.88 and (best_score - second_score) >= 0.04:
        out = [best_p]
        for s, p in scored[1:]:
            if len(out) >= k:
                break
            if s >= 0.86:
                out.append(p)
        return out

    return []


# ---------- API ----------
@app.get("/pb-chat/health")
def health():
    return {"ok": True}


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
            "brand": p.get("brand"),
            "title": p.get("title"),
            "brand_title": p.get("brand_title"),
            "product_type": p.get("product_type"),
            "link": p.get("link"),
        })
    return {"count": len(products), "rows": rows}


@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = ensure_products_loaded()

    # Eğer ürün listesi hala yoksa (feed erişilemez vs)
    if not products:
        return {
            "answer": (
                "Şu an ürün listesini yükleyemedim. Lütfen biraz sonra tekrar dener misiniz?"
            ),
            "products": []
        }

    # 1) brand + keyword typo düzelt
    q_fixed = fix_brand_typo(inp.query)
    q_fixed = fix_keyword_typo(q_fixed)

    hits = simple_search(products, q_fixed, k=6)

    # 2) Hala yoksa konservatif fuzzy
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

    # UI kartları (widget)
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

    # OpenAI yoksa bile UI çalışsın
    if not client:
        return {
            "answer": "Size uygun birkaç öneri hazırladım; aşağıda kartlar halinde görebilirsiniz. İsterseniz yaş ve bütçe aralığını da yazın, seçenekleri daha da daraltayım.",
            "products": ui_products
        }

    # LLM'e sadece bulunan ürünleri veriyoruz (hallucination engeli)
    safe_products = [
        {
            "title": (p.get("brand_title") or "").strip(),
            "price": ((p.get("sale_price") or p.get("price") or "")).strip(),
            "link": add_utm((p.get("link") or "").strip()),
            "product_type": (p.get("product_type") or "").strip(),
            "brand": (p.get("brand") or "").strip(),
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
        "ui_note": "Ürünleri metin olarak listeleme; kartlar ayrı gösterilecek."
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



