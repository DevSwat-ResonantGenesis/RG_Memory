#!/usr/bin/env python3

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

import httpx


def _hdr(user_id: str, org_id: Optional[str], subscription_tier: Optional[str], dev_override: bool) -> Dict[str, str]:
    headers = {"x-user-id": user_id}
    if org_id:
        headers["x-org-id"] = org_id
    if subscription_tier:
        headers["x-subscription-tier"] = subscription_tier
    headers["x-is-dev-override"] = "true" if dev_override else "false"
    return headers


def _post(client: httpx.Client, url: str, headers: Dict[str, str], payload: Dict[str, Any]) -> httpx.Response:
    return client.post(url, headers=headers, json=payload, timeout=20.0)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=os.getenv("MEMORY_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--user-id", required=True)
    parser.add_argument("--org-id", required=True)
    parser.add_argument("--agent-hash", required=True)
    parser.add_argument("--query", default="phase2 smoke")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--seed", default="phase2 smoke")
    parser.add_argument("--include-premium-agent-global", action="store_true")
    parser.add_argument("--dev-override", action="store_true")
    parser.add_argument("--subscription-tier", default=None)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    ingest_url = f"{base_url}/memory/ingest"
    retrieve_url = f"{base_url}/memory/retrieve"

    headers = _hdr(
        user_id=args.user_id,
        org_id=args.org_id,
        subscription_tier=args.subscription_tier,
        dev_override=args.dev_override,
    )

    with httpx.Client() as client:
        # USER GLOBAL
        r1 = _post(
            client,
            ingest_url,
            headers,
            {
                "user_id": args.user_id,
                "org_id": args.org_id,
                "source": "smoke",
                "content": f"[user_global] {args.seed}",
                "metadata": {"kind": "user_global"},
            },
        )
        if r1.status_code >= 300:
            print("ingest user_global failed", r1.status_code, r1.text)
            return 1

        # USER OVERLAY
        r2 = _post(
            client,
            ingest_url,
            headers,
            {
                "user_id": args.user_id,
                "org_id": args.org_id,
                "agent_hash": args.agent_hash,
                "source": "smoke",
                "content": f"[user_overlay] {args.seed}",
                "metadata": {"kind": "user_overlay"},
            },
        )
        if r2.status_code >= 300:
            print("ingest user_overlay failed", r2.status_code, r2.text)
            return 1

        # AGENT GLOBAL PUBLIC
        r3 = _post(
            client,
            ingest_url,
            headers,
            {
                "user_id": None,
                "org_id": args.org_id,
                "agent_hash": args.agent_hash,
                "source": "smoke",
                "content": f"[agent_global public] {args.seed}",
                "metadata": {"kind": "agent_global", "tier": "public"},
            },
        )
        if r3.status_code >= 300:
            print("ingest agent_global public failed", r3.status_code, r3.text)
            return 1

        # AGENT GLOBAL PREMIUM
        r4 = _post(
            client,
            ingest_url,
            headers,
            {
                "user_id": None,
                "org_id": args.org_id,
                "agent_hash": args.agent_hash,
                "source": "smoke",
                "content": f"[agent_global premium] {args.seed}",
                "metadata": {"kind": "agent_global", "tier": "premium"},
            },
        )
        if r4.status_code >= 300:
            print("ingest agent_global premium failed", r4.status_code, r4.text)
            return 1

        # Retrieve without entitlement (unless dev_override/tier passed)
        rr = _post(
            client,
            retrieve_url,
            headers,
            {
                "user_id": args.user_id,
                "org_id": args.org_id,
                "agent_hash": args.agent_hash,
                "query": args.query,
                "limit": args.limit,
                "use_vector_search": True,
            },
        )
        if rr.status_code >= 300:
            print("retrieve failed", rr.status_code, rr.text)
            return 1

        data: List[Dict[str, Any]] = rr.json()

        # Basic assertions
        scopes = [m.get("scope") for m in data]
        tiers = [m.get("tier") for m in data]
        contents = [m.get("content", "") for m in data]

        has_user_global = any("[user_global]" in c for c in contents)
        has_user_overlay = any("[user_overlay]" in c for c in contents)
        has_agent_public = any("[agent_global public]" in c for c in contents)
        has_agent_premium = any("[agent_global premium]" in c for c in contents)

        print(json.dumps({"count": len(data), "scopes": scopes, "tiers": tiers}, indent=2))

        if not has_user_global:
            print("FAIL: missing user_global")
            return 2
        if not has_user_overlay:
            print("FAIL: missing user_overlay")
            return 2
        if not has_agent_public:
            print("FAIL: missing agent_global public")
            return 2

        if args.include_premium_agent_global:
            if not has_agent_premium:
                print("FAIL: expected premium agent_global but not found")
                return 3
        else:
            if has_agent_premium:
                print("FAIL: premium agent_global leaked without entitlement")
                return 3

        print("OK")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
