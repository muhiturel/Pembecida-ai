import os, re, json
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from lxml import etree
from openai import OpenAI

# -------------------------
# App & CORS
# -------------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # MVP: herkese açık. Prod'da pembecida domainleriyle sınırlarız.
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------
# Config
# -------------------------
FEED_URL = os.environ.get("FEED_URL", "").strip()
STORE_PATH = "products.json"

if not FEED_URL:
    # FEED_URL yoksa deploy'da reindex 500 verecek; health yine çalışır.
    pass

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

# -------------------------
# Models
# -------------------------
class ChatIn(BaseModel):
    query: str
    page_url: str | None = None

# -------------------------
# Helpers
# -------------------------
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

def fetch_and_index_feed():
    if not FEED_URL:
        raise RuntimeError("FEED_URL env is missing")

    r = requests.get(FEED_URL, timeout=90)
    r.raise_for_status()

    xml = etree.fromstring(r.content)
    ns = {"g": "http://base.google.com/ns/1.0"}
    items = xml.findall(".//item")

    products = []
    for it in items:
        def get(tag):
            el = it.find(f"g:{tag}", namespaces=ns)
            return (el.text or "").strip() if el is not None else ""

        pid = get("id")
        title = get("title")
        brand = get("brand")
        link = get("link")
        price = get("price")
        sale_price = get("sale_price")
        product_type = get("product_type")
        image = get("image_link")
        add_imgs = [(el.text or "").strip() for el in it.findall("g:additional_image_link", namespaces=ns)]

        # Minimum alanlar
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

def simple_search(products, q, k=6):
    nq = norm(q)
    if not nq:
        return []

    # TR/EN niyet sözlüğü (MVP)
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
        # daha geniş alanlarda ara
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

# -------------------------
# Routes
# -------------------------
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
    products = load_products()
    return {"count": len(products), "sample": products[:limit]}

@app.get("/pb-chat/debug/search")
def debug_search(q: str, k: int = 10):
    products = load_products()
    hits = simple_search(products, q, k=k)
    return {"query": q, "hits": hits}

# -------------------------
# GitHub → app.py → en altlara (debug/sample ve debug/search’in altına) şunu ekle:
# -------------------------

@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = load_products()[:limit]
    out = []
    for p in products:
        out.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "brand": p.get("brand"),
            "brand_title": p.get("brand_title"),
            "product_type": p.get("product_type"),
            "link": p.get("link"),
        })
    return {"count": len(load_products()), "rows": out}



@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()
    hits = simple_search(products, inp.query, k=6)

    # Hit yoksa: ASLA uydurma ürün yok. LLM'e bile gitme.
    if not hits:
        return {
            "answer": (
                "Bu aramada eşleşen ürün bulamadım. Ürünlerimiz orijinaldir. "
                "İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). "
                "İade/kargo gibi konular için: https://www.pembecida.com/sikca-sorulan-sorular"
            ),
            "hits": []
        }

    # LLM'e verilecek güvenli ürün listesi (yalnızca feed'den)
    safe_products = []
    for p in hits:
        safe_products.append({
            "title": (p.get("brand_title") or "").strip(),
            "price": (p.get("sale_price") or p.get("price") or "").strip(),
            "link": (p.get("link") or "").strip(),
            "product_type": (p.get("product_type") or "").strip(),
            "brand": (p.get("brand") or "").strip(),
        })

    system = (
        "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
        "Hedef kitle: çocuğunu sevindirmek isteyen ebeveynler. "
        "İlk yanıtta ürünlerin orijinal olduğunu belirt. "
        "Fiyatlar genelde stabil ama değişmiş olabilir; emin değilsen 'aklımdaki fiyat' deyip ürün sayfasına yönlendir. "
        "İade/kargo vb. sorularda SSS sayfasına yönlendir: https://www.pembecida.com/sikca-sorulan-sorular . "
        "ÇOK ÖNEMLİ: Sana verilen ürün listesinde olmayan hiçbir ürünü ASLA önerme; isim veya link UYDURMA."
    )

    user_payload = {
        "query": inp.query,
        "products": safe_products
    }

    # OpenAI çağrısı
    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)}
        ]
    )

    return {"answer": resp.output_text, "hits": hits[:5]}

@app.get("/pb-chat/widget.js")
def widget():
    # API_BASE sabit: render domain
    js = """
(() => {
  const API_BASE = "https://pembecida-ai.onrender.com";

  const btn = document.createElement("button");
  btn.innerText = "PembeGPT";
  btn.style.cssText = "position:fixed;right:20px;bottom:20px;z-index:99999;padding:10px 14px;border-radius:999px;border:0;cursor:pointer";
  document.body.appendChild(btn);

  const box = document.createElement("div");
  box.style.cssText = "display:none;position:fixed;right:16px;bottom:64px;z-index:99999;width:340px;max-width:calc(100vw - 32px);height:420px;background:#fff;border:1px solid #ddd;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;font-family:system-ui;";
  box.innerHTML = `
    <div style="padding:12px 14px;border-bottom:1px solid #eee;font-weight:600;">Pembecida Asistan</div>
    <div id="pb_msgs" style="padding:10px 14px;height:300px;overflow:auto;font-size:14px;line-height:1.35;"></div>
    <div style="display:flex;gap:8px;padding:10px 10px;border-top:1px solid #eee;">
      <input id="pb_in" placeholder="Ne arıyorsunuz? (örn: 8 yaş hediye, Pop Mart, kalem kutusu...)" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:10px;"/>
      <button id="pb_send" style="padding:10px 12px;border-radius:10px;border:0;cursor:pointer;">Gönder</button>
    </div>
    <div style="padding:8px 14px;font-size:12px;color:#666;border-top:1px solid #f3f3f3;">
      Not: Ürünler orijinaldir. Fiyatlar değişmiş olabilir; en güncel bilgi ürün sayfasındadır.
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
      addMsg("Pembecida", "Merhaba! Ürünlerimiz orijinaldir. Size yardımcı olayım: Ne arıyorsunuz? (İsterseniz yaş / bütçe / kategori de yazabilirsiniz.)");
    }
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
      if (!res.ok) {
        throw new Error((data && (data.detail || data.error)) || ("HTTP " + res.status));
      }

      msgs.lastChild.querySelector("div:last-child").innerHTML =
        (data.answer || "Şu an yanıt oluşturamadım, lütfen tekrar dener misiniz?").replace(/\\n/g, "<br/>");

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

# -------------------------
# Son eklendi
# -------------------------

@app.get("/pb-chat/debug/fields")
def debug_fields(limit: int = 5):
    products = load_products()[:limit]
    out = []
    for p in products:
        out.append({
            "id": p.get("id"),
            "title": p.get("title"),
            "brand": p.get("brand"),
            "brand_title": p.get("brand_title"),
            "product_type": p.get("product_type"),
            "link": p.get("link"),
        })
    return {"count": len(load_products()), "rows": out}



