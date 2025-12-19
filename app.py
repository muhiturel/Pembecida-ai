import os, re, json
from fastapi import FastAPI
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import requests
from lxml import etree
from openai import OpenAI

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

FEED_URL = os.environ.get("FEED_URL", "")
STORE_PATH = "products.json"
WIDGET_JS = None  # Render URL deploy sonrası set edeceğiz

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

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
    with open(STORE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

def save_products(products):
    with open(STORE_PATH, "w", encoding="utf-8") as f:
        json.dump(products, f, ensure_ascii=False)

def fetch_and_index_feed():
    r = requests.get(FEED_URL, timeout=90)
    r.raise_for_status()
    xml = etree.fromstring(r.content)
    ns = {"g":"http://base.google.com/ns/1.0"}
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
        add_imgs = [ (el.text or "").strip() for el in it.findall("g:additional_image_link", namespaces=ns) ]

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

@app.get("/pb-chat/health")
def health():
    return {"ok": True}

@app.post("/pb-chat/reindex")
def reindex():
    fetch_and_index_feed()
    return {"ok": True, "count": len(load_products())}

def simple_search(products, q, k=6):
    nq = norm(q)
    if not nq:
        return []
    tokens = nq.split()
    scored = []
    for p in products:
        hay = norm(" ".join([p.get("brand_title",""), p.get("product_type","")]))
        score = sum(1 for t in tokens if t in hay)
        if score > 0:
            scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:k]]

@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()
    hits = simple_search(products, inp.query, k=6)

    cards = []
    for p in hits:
        price = p.get("sale_price") or p.get("price") or ""
        cards.append(f"- {p.get('brand_title')} | {price} | {p.get('link')}")

    system = (
      "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
      "Hedef kitle: çocuğunu sevindirmek isteyen ebeveynler. "
      "İlk yanıtta ürünlerin orijinal olduğunu belirt. "
      "Fiyatlar genelde stabil ama değişmiş olabilir; emin değilsen 'aklımdaki fiyat' deyip ürün sayfasına yönlendir. "
      "İade/kargo vb. sorularda SSS sayfasına yönlendir: https://www.pembecida.com/sikca-sorulan-sorular ."
    )

    user = (
      f"Kullanıcı sorusu: {inp.query}\n\n"
      f"Bulduğum en alakalı ürünler:\n" + ("\n".join(cards) if cards else "Uygun ürün eşleşmesi bulamadım.") +
      "\n\nYanıtı kısa, net ve alışverişe yönlendiren şekilde ver. "
      "Uygun 2-5 ürün öner, her birini 1 cümleyle özetle ve link ver."
    )

    resp = client.responses.create(
        model="gpt-4.1-mini",
        input=[
            {"role":"system","content":system},
            {"role":"user","content":user}
        ]
    )

    return {"answer": resp.output_text, "hits": hits[:5]}

@app.get("/pb-chat/widget.js")
def widget():
    # Render URL'i üzerinden API_BASE'i dinamik basıyoruz:
    # request.host Render domainini verir.
    # Basit tutuyoruz.
    # Not: http/https render tarafında https olacak.
    js = """
(() => {
  const API_BASE = location.origin.replace('www.', ''); // render domain
  const btn = document.createElement("button");
  btn.innerText = "Yardım";
  btn.style.cssText = "position:fixed;right:16px;bottom:16px;z-index:99999;padding:10px 14px;border-radius:999px;border:0;cursor:pointer;";
  document.body.appendChild(btn);

  const box = document.createElement("div");
  box.style.cssText = "display:none;position:fixed;right:16px;bottom:64px;z-index:99999;width:340px;max-width:calc(100vw - 32px);height:420px;background:#fff;border:1px solid #ddd;border-radius:16px;box-shadow:0 8px 24px rgba(0,0,0,.12);overflow:hidden;font-family:system-ui;";
  box.innerHTML = `
    <div style="padding:12px 14px;border-bottom:1px solid #eee;font-weight:600;">Pembecida Asistan</div>
    <div id="pb_msgs" style="padding:10px 14px;height:300px;overflow:auto;font-size:14px;line-height:1.35;"></div>
    <div style="display:flex;gap:8px;padding:10px 10px;border-top:1px solid #eee;">
      <input id="pb_in" placeholder="Ne arıyorsunuz? (örn: 8 yaş hediye, Pop Mart, sırt çantası...)" style="flex:1;padding:10px;border:1px solid #ddd;border-radius:10px;"/>
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
      addMsg("Pembecida", "Merhaba! Ürünlerimiz orijinaldir. Size yardımcı olayım: Kimin için hediye arıyorsunuz, yaş ve bütçe aralığı var mı?");
    }
  };

  const doSend = async () => {
    const q = input.value.trim();
    if (!q) return;
    input.value = "";
    addMsg("Siz", q);
    addMsg("Pembecida", "Hemen bakıyorum…");

    const res = await fetch(API_BASE + "/pb-chat/chat", {
      method: "POST",
      headers: {"Content-Type":"application/json"},
      body: JSON.stringify({ query: q, page_url: location.href })
    });

    const data = await res.json();
    msgs.lastChild.querySelector("div:last-child").innerHTML = (data.answer || "").replace(/\\n/g, "<br/>");
  };

  send.onclick = doSend;
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") doSend(); });
})();
""".strip()
    return Response(js, media_type="application/javascript")
