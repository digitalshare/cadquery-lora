"""Minimal Fireworks REST helper (datasets, SFT jobs, deployments).

  python scripts/fw_api.py dataset-create <id> <jsonl>
  python scripts/fw_api.py sftj-create <json-file-with-job-spec> [job-id]
  python scripts/fw_api.py sftj-get <job-id>
  python scripts/fw_api.py deploy <model> <deployment-id> <shape|default> [min] [max]
  python scripts/fw_api.py deploy-get <deployment-id>
  python scripts/fw_api.py deploy-delete <deployment-id>
"""
import json, os, sys, time, urllib.request, urllib.error

ACCT = os.environ.get("FIREWORKS_ACCOUNT", "wjing63")
KEY = os.environ["FIREWORKS_API_KEY"]
BASE = f"https://api.fireworks.ai/v1/accounts/{ACCT}"

def req(method, url, body=None, raw=None, headers=None):
    h = {"Authorization": f"Bearer {KEY}"}
    data = None
    if body is not None:
        h["Content-Type"] = "application/json"; data = json.dumps(body).encode()
    if raw is not None:
        data = raw
    if headers: h.update(headers)
    r = urllib.request.Request(url, data=data, method=method, headers=h)
    try:
        with urllib.request.urlopen(r, timeout=300) as resp:
            t = resp.read()
            return json.loads(t) if t.strip().startswith(b"{") else t
    except urllib.error.HTTPError as e:
        raise SystemExit(f"HTTP {e.code} {method} {url}\n{e.read().decode()[:1500]}")

def dataset_create(ds_id, path):
    n = sum(1 for _ in open(path))
    out = req("POST", f"{BASE}/datasets", {"datasetId": ds_id, "dataset": {"userUploaded": {}, "exampleCount": n}})
    print("created", out.get("name"), "examples", n)
    size = os.path.getsize(path)
    up = req("POST", f"{BASE}/datasets/{ds_id}:getUploadEndpoint", {"filenameToSize": {os.path.basename(path): size}})
    url = up["filenameToSignedUrls"][os.path.basename(path)]
    with open(path, "rb") as f:
        r = urllib.request.Request(url, data=f.read(), method="PUT", headers={"Content-Type": "application/octet-stream", "x-goog-content-length-range": f"0,{size}"})
        urllib.request.urlopen(r, timeout=600).read()
    val = req("POST", f"{BASE}/datasets/{ds_id}:validateUpload", {})
    for _ in range(60):
        d = req("GET", f"{BASE}/datasets/{ds_id}")
        print("state", d.get("state"), d.get("status", {}).get("message", "") or "")
        if d.get("state") in ("READY", "FAILED"): break
        time.sleep(5)
    return d

def sftj_create(spec_path, job_id=None):
    spec = json.load(open(spec_path))
    url = f"{BASE}/supervisedFineTuningJobs" + (f"?supervisedFineTuningJobId={job_id}" if job_id else "")
    out = req("POST", url, spec)
    print(json.dumps(out, indent=2)[:3000]); return out

def deploy(model, dep_id, shape, mn=1, mx=1):
    body = {"baseModel": model, "minReplicaCount": int(mn), "maxReplicaCount": int(mx), "displayName": dep_id}
    if shape != "default":
        body["deploymentShape"] = shape if shape.startswith("accounts/") else f"accounts/fireworks/deploymentShapes/{shape}"
    out = req("POST", f"{BASE}/deployments?deploymentId={dep_id}", body)
    print(json.dumps(out, indent=2)[:2000]); return out

def main():
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "dataset-create": dataset_create(*args)
    elif cmd == "sftj-create": sftj_create(*args)
    elif cmd == "sftj-get":
        d = req("GET", f"{BASE}/supervisedFineTuningJobs/{args[0]}"); print(json.dumps({k: d.get(k) for k in ["name","state","status","jobProgress","outputModel","createTime","completedTime","estimatedCost"]}, indent=2))
    elif cmd == "deploy": deploy(*args)
    elif cmd == "deploy-get":
        d = req("GET", f"{BASE}/deployments/{args[0]}"); print(json.dumps({k: d.get(k) for k in ["name","state","status","baseModel","deploymentShape","replicaCount","minReplicaCount","maxReplicaCount"]}, indent=2))
    elif cmd == "deploy-delete":
        print(req("DELETE", f"{BASE}/deployments/{args[0]}"))
    elif cmd == "get": print(json.dumps(req("GET", f"{BASE}/{args[0]}"), indent=2)[:4000])
    else: raise SystemExit(__doc__)

if __name__ == "__main__":
    main()
