"""Sales platforms: Stripe (fully automatic), Whop (automatic, API v1), Gumroad (sales sync + best-effort create).

Every client has the same ``publish`` call and returns a ``ListingResult``. Webhook parsing turns each
platform's payload into a ``SaleEvent`` so the rest of the office never sees platform details.

* Stripe: Product -> Price -> Payment Link via the REST API (no SDK needed). Webhook signatures are
  verified with the documented HMAC-SHA256 scheme.
* Whop: POST /api/v1/products then /api/v1/plans (one_time). Endpoints follow Whop's v1 docs; check them
  against https://docs.whop.com if Whop changes the API.
* Gumroad: its public API documents sales, product enable/disable and webhooks ("resource subscriptions").
  Product creation (POST /v2/products) is not in every version of the docs, so it is tried once and, on
  failure, the listing becomes "manual": the CEO gets the ready-made title/description/price/file to upload.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import time
from dataclasses import dataclass
from typing import Any

import httpx

from app.config import Settings, secret


@dataclass(frozen=True)
class ListingResult:
    platform: str
    external_id: str
    url: str
    status: str  # live | manual | error
    note: str = ""


@dataclass(frozen=True)
class SaleEvent:
    platform: str
    external_id: str
    gross_cents: int
    buyer_email: str
    listing_ref: str  # payment link id / gumroad product id / whop product or plan id
    fee_cents: int | None = None  # None = estimate from the platform's fee schedule
    refunded: bool = False
    payment_ref: str = ""  # payment intent / charge id (refunds and disputes are matched on it)
    disputed: bool = False


@dataclass(frozen=True)
class CheckoutSignal:
    """A checkout that did not turn into a sale: abandoned (session expired) or payment failed."""
    kind: str  # abandoned | failed
    external_id: str
    email: str
    listing_ref: str
    consent: bool  # buyer opted in to follow-up emails at checkout
    reason: str = ""


class WebhookError(ValueError):
    pass


def estimate_fee(s: Settings, platform: str, gross_cents: int) -> int:
    if platform == "stripe":
        pct = s.stripe_fee_pct + s.intl_share * s.stripe_intl_card_pct  # expected mix of domestic/foreign cards
        return round(gross_cents * pct + s.stripe_fee_fixed_usd * 100)
    if platform == "gumroad":
        return round(gross_cents * s.gumroad_fee_pct + s.gumroad_fee_fixed_usd * 100)
    if platform == "whop":
        return round(gross_cents * s.whop_fee_pct)
    return 0


# ============================================================================ Stripe
class StripeClient:
    API = "https://api.stripe.com/v1"

    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client

    async def _post(self, path: str, data: dict[str, Any]) -> dict[str, Any]:
        r = await self.client.post(f"{self.API}/{path}", data=data, auth=(secret(self.s.stripe_secret_key), ""))
        if r.status_code >= 400:
            raise RuntimeError(f"stripe {path}: HTTP {r.status_code} {r.text[:300]}")
        return dict(r.json())

    async def publish(self, product_id: int, name: str, description: str, price_cents: int) -> ListingResult:
        prod = await self._post("products", {"name": name, "description": description[:800],
                                             "metadata[studio_product_id]": str(product_id)})
        price = await self._post("prices", {"product": prod["id"], "unit_amount": price_cents, "currency": "usd"})
        link = await self._post("payment_links", {
            "line_items[0][price]": price["id"], "line_items[0][quantity]": 1,
            "after_completion[type]": "redirect",
            "after_completion[redirect][url]": f"{self.s.public_base_url}/thanks?session_id={{CHECKOUT_SESSION_ID}}",
            "metadata[studio_product_id]": str(product_id),
            # asks for marketing consent where required, so abandoned checkouts can get one recovery email
            "consent_collection[promotions]": "auto",
            "allow_promotion_codes": "true",
            # CRO: minimal fields; wallets (Apple Pay / Google Pay) show automatically when enabled in the Stripe dashboard
            "billing_address_collection": "auto",
            "submit_type": "pay",
        })
        return ListingResult("stripe", link["id"], link["url"], "live", f"product {prod['id']} price {price['id']}")

    async def launch_promo(self, slug: str, percent: int, hours: int = 48) -> dict[str, Any]:
        """Real scarcity: a coupon that truly expires after ``hours``."""
        until = int(time.time()) + hours * 3600
        coupon = await self._post("coupons", {"percent_off": percent, "duration": "once", "redeem_by": until,
                                              "name": f"Launch {percent}% off"})
        code = ("LAUNCH" + re.sub(r"[^A-Z0-9]", "", slug.upper()))[:20]
        promo = await self._post("promotion_codes", {"coupon": coupon["id"], "code": code, "expires_at": until})
        return {"code": promo.get("code", code), "percent": percent, "expires_at": until, "coupon": coupon["id"]}

    async def reprice(self, product_ref: str, old_link: str, product_id: int, price_cents: int) -> ListingResult:
        """Stripe payment links cannot change price: new Price + new link, old link deactivated."""
        price = await self._post("prices", {"product": product_ref, "unit_amount": price_cents, "currency": "usd"})
        link = await self._post("payment_links", {
            "line_items[0][price]": price["id"], "line_items[0][quantity]": 1, "after_completion[type]": "redirect",
            "after_completion[redirect][url]": f"{self.s.public_base_url}/thanks?session_id={{CHECKOUT_SESSION_ID}}",
            "metadata[studio_product_id]": str(product_id), "consent_collection[promotions]": "auto",
            "allow_promotion_codes": "true", "billing_address_collection": "auto", "submit_type": "pay"})
        await self.deactivate(old_link)
        return ListingResult("stripe", link["id"], link["url"], "live", f"product {product_ref} price {price['id']}")

    async def settlement(self, payment_intent: str) -> dict[str, Any] | None:
        """Actual fee and net in the settlement currency (Nina's audit), from the balance transaction."""
        r = await self.client.get(f"{self.API}/payment_intents/{payment_intent}",
                                  params={"expand[]": "latest_charge.balance_transaction"}, auth=(secret(self.s.stripe_secret_key), ""))
        if r.status_code != 200:
            return None
        bt = ((r.json().get("latest_charge") or {}).get("balance_transaction")) or {}
        if not isinstance(bt, dict) or "fee" not in bt:
            return None
        rate = float(bt.get("exchange_rate") or 1.0)
        return {"fee_usd_cents": round(int(bt["fee"]) / rate), "net": int(bt.get("net", 0)), "currency": bt.get("currency", "usd")}

    async def deactivate(self, link_id: str) -> None:
        await self._post(f"payment_links/{link_id}", {"active": "false"})

    # ------------------------------------------------------------------ self-configuration (Leo / Nina)
    async def _get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        r = await self.client.get(f"{self.API}/{path}", params=params or {}, auth=(secret(self.s.stripe_secret_key), ""))
        if r.status_code >= 400:
            raise RuntimeError(f"stripe {path}: HTTP {r.status_code} {r.text[:300]}")
        return dict(r.json())

    async def account(self) -> dict[str, Any]:
        """Country and default (payout) currency of the connected Stripe account."""
        a = await self._get("account")
        return {"country": a.get("country", ""), "default_currency": (a.get("default_currency") or "usd").lower()}

    async def balance(self) -> list[dict[str, Any]]:
        """Available + pending balance per currency (minor units)."""
        b = await self._get("balance")
        out: dict[str, int] = {}
        for part in ("available", "pending"):
            for row in b.get(part, []):
                out[row["currency"]] = out.get(row["currency"], 0) + int(row.get("amount", 0))
        return [{"currency": c, "amount": a} for c, a in out.items()]

    async def ensure_webhook(self, url: str, events: list[str], known_secret: str) -> dict[str, Any]:
        """Create (or fix) the endpoint that receives payment events. Stripe returns the signing secret only on create,
        so an endpoint whose secret we don't know is replaced by a new one."""
        endpoints = (await self._get("webhook_endpoints", {"limit": 100})).get("data", [])
        mine = [e for e in endpoints if e.get("url") == url]
        if mine and known_secret:
            ep = mine[0]
            if set(events) - set(ep.get("enabled_events", [])) or ep.get("status") != "enabled":
                await self._post(f"webhook_endpoints/{ep['id']}", {**{f"enabled_events[{i}]": e for i, e in enumerate(events)},
                                                                    "disabled": "false"})
                return {"id": ep["id"], "secret": None, "action": "updated events"}
            return {"id": ep["id"], "secret": None, "action": "already correct"}
        for ep in mine:  # unknown secret: replace
            r = await self.client.delete(f"{self.API}/webhook_endpoints/{ep['id']}", auth=(secret(self.s.stripe_secret_key), ""))
            if r.status_code >= 400:
                raise RuntimeError(f"stripe delete webhook: HTTP {r.status_code}")
        ep = await self._post("webhook_endpoints", {"url": url, "description": "Product Studio (managed by Leo)",
                                                    **{f"enabled_events[{i}]": e for i, e in enumerate(events)}})
        return {"id": ep["id"], "secret": ep.get("secret"), "action": "created"}

    async def enable_wallets(self) -> dict[str, Any]:
        """Turn on Apple Pay and Google Pay in the default payment method configuration."""
        confs = (await self._get("payment_method_configurations", {"limit": 20})).get("data", [])
        conf = next((c for c in confs if c.get("is_default")), confs[0] if confs else None)
        if not conf:
            return {"action": "no payment method configuration found"}
        state = {w: ((conf.get(w) or {}).get("display_preference") or {}).get("value") for w in ("apple_pay", "google_pay")}
        if all(v == "on" for v in state.values()):
            return {"action": "already on", **state}
        await self._post(f"payment_method_configurations/{conf['id']}", {
            "apple_pay[display_preference][preference]": "on", "google_pay[display_preference][preference]": "on"})
        return {"action": "turned on", "configuration": conf["id"]}


def verify_stripe_signature(payload: bytes, header: str, secret_key: str, tolerance: int = 300) -> None:
    parts: dict[str, list[str]] = {}
    for item in header.split(","):
        k, _, v = item.partition("=")
        parts.setdefault(k.strip(), []).append(v.strip())
    try:
        ts = int(parts["t"][0])
    except (KeyError, ValueError) as exc:
        raise WebhookError("missing timestamp") from exc
    if abs(time.time() - ts) > tolerance:
        raise WebhookError("timestamp outside tolerance")
    expected = hmac.new(secret_key.encode(), f"{ts}.".encode() + payload, hashlib.sha256).hexdigest()
    if not any(hmac.compare_digest(expected, sig) for sig in parts.get("v1", [])):
        raise WebhookError("bad signature")


def parse_stripe_event(event: dict[str, Any]) -> SaleEvent | None:
    obj = event.get("data", {}).get("object", {})
    if event.get("type") == "checkout.session.completed" and obj.get("payment_status") == "paid":
        return SaleEvent("stripe", obj["id"], int(obj.get("amount_total") or 0),
                         (obj.get("customer_details") or {}).get("email") or "", obj.get("payment_link") or "",
                         payment_ref=obj.get("payment_intent") or "")
    if event.get("type") == "charge.refunded":
        pi = obj.get("payment_intent") or obj["id"]
        return SaleEvent("stripe", pi, int(obj.get("amount") or 0), "", "", refunded=True, payment_ref=pi)
    if event.get("type") == "charge.dispute.created":
        pi = obj.get("payment_intent") or obj.get("charge") or obj.get("id", "")
        return SaleEvent("stripe", pi, int(obj.get("amount") or 0), "", "", refunded=True, payment_ref=pi, disputed=True)
    return None


def parse_stripe_checkout_signal(event: dict[str, Any]) -> CheckoutSignal | None:
    obj = event.get("data", {}).get("object", {})
    if event.get("type") == "checkout.session.expired":
        consent = (obj.get("consent") or {}).get("promotions") == "opt_in"
        return CheckoutSignal("abandoned", obj.get("id", ""), (obj.get("customer_details") or {}).get("email") or "",
                              obj.get("payment_link") or "", consent)
    if event.get("type") == "payment_intent.payment_failed":
        err = (obj.get("last_payment_error") or {}).get("code") or "unknown"
        return CheckoutSignal("failed", obj.get("id", ""), obj.get("receipt_email") or "", "", False, err)
    return None


# ============================================================================ Gumroad
class GumroadClient:
    API = "https://api.gumroad.com/v2"

    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client

    @property
    def _token(self) -> dict[str, str]:
        return {"access_token": secret(self.s.gumroad_access_token)}

    async def publish(self, product_id: int, name: str, description: str, price_cents: int) -> ListingResult:
        r = await self.client.post(f"{self.API}/products", data={**self._token, "name": name, "price": price_cents,
                                                                 "description": description, "native_type": "digital"})
        if r.status_code == 200 and r.json().get("success"):
            p = r.json()["product"]
            await self.client.put(f"{self.API}/products/{p['id']}/enable", data=self._token)
            return ListingResult("gumroad", p["id"], p.get("short_url", ""), "live",
                                 "created via API; upload the .zip as the product file on Gumroad")
        return ListingResult("gumroad", "", "https://gumroad.com/products/new", "manual",
                             f"API create unavailable (HTTP {r.status_code}); upload manually with the prepared listing")

    async def subscribe_sales(self, key: str = "") -> None:
        url = f"{self.s.public_base_url}/webhooks/gumroad?key={key or secret(self.s.gumroad_webhook_secret)}"
        await self.client.put(f"{self.API}/resource_subscriptions",
                              data={**self._token, "resource_name": "sale", "post_url": url})

    async def recent_sales(self, after: str) -> list[SaleEvent]:
        r = await self.client.get(f"{self.API}/sales", params={**self._token, "after": after})
        r.raise_for_status()
        return [SaleEvent("gumroad", s["id"], int(s.get("price") or 0), s.get("email", ""), s.get("product_id", ""),
                          refunded=bool(s.get("refunded"))) for s in r.json().get("sales", [])]


def parse_gumroad_ping(form: dict[str, str]) -> SaleEvent:
    # Gumroad "ping" is form-encoded; price is in cents. Not signed: the URL carries our secret key.
    return SaleEvent("gumroad", form.get("sale_id", ""), int(form.get("price") or 0), form.get("email", ""),
                     form.get("product_id", ""), refunded=form.get("refunded") == "true")


# ============================================================================ Whop
class WhopClient:
    API = "https://api.whop.com/api/v1"

    def __init__(self, s: Settings, client: httpx.AsyncClient) -> None:
        self.s, self.client = s, client

    async def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        r = await self.client.post(f"{self.API}/{path}", json=body,
                                   headers={"Authorization": f"Bearer {secret(self.s.whop_api_key)}"})
        if r.status_code >= 400:
            raise RuntimeError(f"whop {path}: HTTP {r.status_code} {r.text[:300]}")
        return dict(r.json())

    async def publish(self, product_id: int, name: str, description: str, price_cents: int) -> ListingResult:
        prod = await self._post("products", {"company_id": self.s.whop_company_id, "title": name[:80],
                                             "description": description[:1000]})
        plan = await self._post("plans", {"company_id": self.s.whop_company_id, "product_id": prod["id"],
                                          "plan_type": "one_time", "initial_price": price_cents / 100, "currency": "usd"})
        url = plan.get("purchase_url") or plan.get("direct_link") or ""
        return ListingResult("whop", str(prod["id"]), url, "live", f"plan {plan.get('id')}")


def verify_hmac_hex(payload: bytes, signature: str, secret_key: str) -> None:
    expected = hmac.new(secret_key.encode(), payload, hashlib.sha256).hexdigest()
    sig = signature.split("=", 1)[-1].strip()
    if not hmac.compare_digest(expected, sig):
        raise WebhookError("bad signature")


def parse_whop_event(event: dict[str, Any]) -> SaleEvent | None:
    action = event.get("action") or event.get("type") or ""
    d = event.get("data", {})
    if action in ("payment.succeeded", "payment_succeeded"):
        amount = d.get("final_amount") or d.get("total") or d.get("subtotal") or 0
        user = d.get("user") or {}
        product = d.get("product") or {}
        return SaleEvent("whop", str(d.get("id")), round(float(amount) * 100), user.get("email", ""),
                         str(product.get("id") or d.get("product_id") or ""))
    return None
