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

FEED_URL = os.environ.get("FEED_URL", "").strip()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STORE_PATH = os.path.join(BASE_DIR, "products.json")

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", ""))

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
            # 1) g:tag
            el = it.find(f"g:{tag}", namespaces=ns)
            if el is not None and (el.text or "").strip():
                return (el.text or "").strip()
            # 2) fallback: namespacesiz
            el2 = it.find(tag)
            if el2 is not None and (el2.text or "").strip():
                return (el2.text or "").strip()
            return ""

        pid = get("id")
        title = get("title")
        brand = get("brand")
        link = get("link")
        price = get("price")
        sale_price = get("sale_price")
        product_type = get("product_type")
        image = get("image_link")
        add_imgs = [(el.text or "").strip() for el in it.findall("g:additional_image_link", namespaces=ns)] \
                   or [(el.text or "").strip() for el in it.findall("additional_image_link")]

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

# smiple search içerisine fuzzytypo_suggest ekleme başı
from difflib import SequenceMatcher

def fuzzy_typo_suggest(products, q: str, k: int = 3):
    """
    Çok konservatif typo düzeltme:
    - Sadece çok yüksek benzerlikte öneri döner
    - Tek ve net kazanan yoksa boş döner
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
        ]))
        if not hay:
            continue

        # SequenceMatcher 0..1 arası benzerlik döner
        r = SequenceMatcher(None, nq, hay).ratio()
        scored.append((r, p))

    if not scored:
        return []

    scored.sort(key=lambda x: x[0], reverse=True)

    best_score, best_p = scored[0]
    second_score = scored[1][0] if len(scored) > 1 else 0.0

    # ✅ Çok katı eşik: typo ise ancak gerçekten çok yakınsa öner
    # - best >= 0.88 (yüksek benzerlik)
    # - best ile second arasında en az 0.04 fark olsun (tek net kazanan)
    if best_score >= 0.88 and (best_score - second_score) >= 0.04:
        # benzer birkaç ürünü de göstermek için en üsttekilerden filtreleyebiliriz
        out = [best_p]
        # İstersen 2-3 tane daha ekleyelim ama benzerlik eşiği yüksek kalsın
        for s, p in scored[1:]:
            if len(out) >= k:
                break
            if s >= 0.86:
                out.append(p)
        return out
    
    # brand typo düzeltme başla
    def fix_brand_typo(q: str) -> str:
        """
        Kısa/tek kelime typo'larda marka düzeltme (çok konservatif).
        Örn: smgle -> smiggle
        """
        nq = norm(q)
        if not nq:
            return q

        tokens = nq.split()

    # Sadece kısa/az token'lı sorgularda çalışsın (risk azaltma)
        if len(tokens) > 3:
            return q

        known = ["smiggle", "pop mart", "popmart", "pembecida"]
    # Token bazlı karşılaştırma
        new_tokens = []
        changed = False

        for t in tokens:
            best = (0.0, None)
        for k in known:
            # Çok kelimeli marka için (pop mart) token'ı birleştirmeye çalışmayalım;
            # burada kısa token'larda asıl hedef "smiggle" gibi tek kelime.
            if " " in k:
                continue
            r = SequenceMatcher(None, t, k).ratio()
            if r > best[0]:
                best = (r, k)

        # Kısa token için daha düşük ama halen katı eşik
        # smgle vs smiggle ~ 0.83-0.86 aralığına gelir
        if best[1] and best[0] >= 0.82 and len(t) >= 4:
            new_tokens.append(best[1])
            changed = True
        else:
            new_tokens.append(t)

        if changed:
            return " ".join(new_tokens)
        return q
        # brand typo düzeltme son
    
    # Şüphe varsa: hiçbir şey döndürme
    return []
# smiple search içerisine fuzzytypo_suggest ekleme sonu

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

@app.get("/pb-chat/debug/search")
def debug_search(q: str, k: int = 10):
    products = load_products()
    hits = simple_search(products, q, k=k)
    return {"query": q, "hits": hits}

@app.post("/pb-chat/chat")
def chat(inp: ChatIn):
    products = load_products()

# products.json boşsa (deploy sonrası sık olur) otomatik reindex dene
    if not products:
        try:
            fetch_and_index_feed()
        except Exception:
            pass
        products = load_products()

    # 1) brand typo düzelt (smgle -> smiggle gibi)
    q_fixed = fix_brand_typo(inp.query)

    hits = simple_search(products, q_fixed, k=6)

    # 2) hala yoksa, konservatif genel typo önerisi dene
    if not hits:
        typo_hits = fuzzy_typo_suggest(products, inp.query, k=3)
    if typo_hits:
        hits = typo_hits
    else:
        return {
            "answer": (
                'Bu aramada eşleşen ürün bulamadım. '
                'İsterseniz marka + ürün tipiyle arayabilirsiniz (örn: “Smiggle kalem kutusu”). '
                'İade/kargo gibi konular için <a href="https://www.pembecida.com/sikca-sorulan-sorular?utm_source=pembegpt&utm_medium=chatbot&utm_campaign=pembecida">Sıkça Sorulan Sorular</a> sayfamızı inceleyebilirsiniz.'
            ),
            "products": []
        }

    top = hits[:5]
    ui_products = []
    for p in top:
        ui_products.append({
            "title": p.get("brand_title") or p.get("title") or "",
            "link": p.get("link") or "",
            "price": (p.get("sale_price") or p.get("price") or ""),
            "image": p.get("image_link") or "",
        })

    safe_products = []
    for p in top:
        safe_products.append({
            "title": (p.get("brand_title") or "").strip(),
            "price": ((p.get("sale_price") or p.get("price") or "")).strip(),
            "link": (p.get("link") or "").strip(),
            "product_type": (p.get("product_type") or "").strip(),
            "brand": (p.get("brand") or "").strip(),
        })

    system = (
    "Sen Pembecida'nın site içi ürün asistanısın. Kullanıcıyla SİZ diye konuş, sıcak ve samimi ol. "
    "Hedef kitle: çocuğunu sevindirmek isteyen ebeveynler. "
    "İlk yanıtta ürünlerin orijinal olduğunu belirt. "
    "Bazı yazılmalarda İstanbul Zorlu Center mağazası olduğunu belirt ve özellikle orijinal olup olmadığı gibi bir soru gelirse mağazada görebileceğini söyle."
    "Eğer Pop Mart markalı Bling Box/sürpriz kutu ürünü soruluyorsa ürünlerin değişim veya iade alınamayacağını yaz."
    "İade/kargo vb. sorularda SSS sayfasına yönlendir: https://www.pembecida.com/sikca-sorulan-sorular . "
    "ÇOK ÖNEMLİ: Sana verilen ürün listesinde olmayan hiçbir ürünü ASLA önerme; isim veya link UYDURMA. "
    "ÇIKTI KURALI: SADECE 2-3 cümlelik kısa bir karşılama ve yönlendirme yaz. "
    "Ürünleri metin olarak listeleme, link yazma, madde madde ürün yazma. "
    "Ürün önerileri aşağıda kart olarak gösterilecek."
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

    return {"answer": resp.output_text, "products": ui_products}

@app.get("/pb-chat/widget.js")
def widget():
    js = r"""
(() => {
  const API_BASE = "https://pembecida-ai.onrender.com";

  // ---- Styles (link + button ring + mobile rules)
  const style = document.createElement("style");
  style.innerHTML = `
    #pb_msgs a { color:#0645AD; text-decoration:underline; }
    #pb_msgs a:visited { color:#0b0080; }

    /* Wrapper: fixed container for the button */
    .pb-gpt-wrap{
      position:fixed;
      right:16px;
      bottom:16px;
      z-index:99999;
    }

    /* Animated gradient ring */
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

    /* Button itself */
    .pb-gpt-btn{
      position:relative;
      z-index:2;
      padding:20px 28px;          /* ~2x feel */
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

    /* Mobile: center horizontally + scale to 1.7 instead of 2.0 => 0.85 */
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

  // ---- Button Wrapper + Button
  const btnWrap = document.createElement("div");
  btnWrap.className = "pb-gpt-wrap";

  const btn = document.createElement("button");
  btn.className = "pb-gpt-btn";
  btn.innerText = "PembeGPT";

  btnWrap.appendChild(btn);
  document.body.appendChild(btnWrap);

  // ---- Chat Box
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
      const addUtm = (url) => {
          try {
            const u = new URL(url, location.origin);
            u.searchParams.set("utm_source", "pembegpt");
            u.searchParams.set("utm_medium", "chatbot");
            u.searchParams.set("utm_campaign", "pembecida");
            return u.toString();
          } catch {
            return url;
          }
        };

        const link = addUtm(p.link || "#");

      const img = p.image || "";
      return `
        <div style="display:flex;gap:10px;padding:10px;border:1px solid #eee;border-radius:12px;margin-top:10px;">
          ${img ? `<img src="${img}" alt="${title}" style="width:64px;height:64px;object-fit:cover;border-radius:10px;border:1px solid #eee;">` : ``}
          <div style="flex:1;min-width:0;">
            <div style="font-weight:600;font-size:14px;line-height:1.25;margin-bottom:4px;">${title}</div>
            ${price ? `<div style="font-size:13px;color:#333;margin-bottom:8px;">${price}</div>` : ``}
            <a href="${link}"
               style="display:inline-block;text-decoration:none;padding:8px 10px;border-radius:10px;border:1px solid #ddd;font-size:13px;">
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







