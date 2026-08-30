import time
import json
import urllib.request
import urllib.error
import sys

BASE_URL = "http://127.0.0.1:8001"

def make_request(url, method="GET", body=None):
    headers = {"Content-Type": "application/json"}
    data = json.dumps(body).encode("utf-8") if body else None
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req) as resp:
            status_code = resp.getcode()
            response_data = json.loads(resp.read().decode("utf-8"))
            return status_code, response_data
    except urllib.error.HTTPError as e:
        status_code = e.code
        try:
            response_data = json.loads(e.read().decode("utf-8"))
        except Exception:
            response_data = e.reason
        return status_code, response_data
    except urllib.error.URLError as e:
        return 500, {"error": str(e)}

def run_tests():
    print("============================================================")
    print("GraphGuard AI Backend Automated API Verification")
    print("============================================================")

    passed_tests = 0
    total_tests = 0

    # 1. GET /api/health
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/health")
    if code == 200 and res.get("status") == "healthy" and res.get("model_loaded") and res.get("transactions_loaded"):
        print("[PASS] 1. GET /api/health:", res)
        passed_tests += 1
    else:
        print("[FAIL] 1. GET /api/health:", code, res)

    # 2. GET /api/dashboard
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/dashboard")
    if code == 200 and res.get("total_transactions") == 203769 and res.get("total_edges") == 234355:
        print(f"[PASS] 2. GET /api/dashboard: Total Tx={res['total_transactions']}, Edges={res['total_edges']}, Illicit={res['illicit_transactions']}")
        passed_tests += 1
    else:
        print("[FAIL] 2. GET /api/dashboard:", code, res)

    # 3. GET /api/transactions
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/transactions?page=1&page_size=5")
    if code == 200 and "transactions" in res and len(res["transactions"]) == 5:
        print(f"[PASS] 3. GET /api/transactions: Retracted {len(res['transactions'])} transactions (Total={res['total']})")
        passed_tests += 1
    else:
        print("[FAIL] 3. GET /api/transactions:", code, res)

    # Pick a real transaction ID from pagination results
    sample_tx_id = res["transactions"][0]["tx_id"]

    # 4. GET /api/transactions/{tx_id}
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/transactions/{sample_tx_id}")
    if code == 200 and res.get("tx_id") == sample_tx_id:
        print(f"[PASS] 4. GET /api/transactions/{sample_tx_id}: Class='{res['class']}', RiskLevel='{res['risk_level']}'")
        passed_tests += 1
    else:
        print(f"[FAIL] 4. GET /api/transactions/{sample_tx_id}:", code, res)

    # 5. GET /api/transactions/{tx_id}/neighbors
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/transactions/{sample_tx_id}/neighbors")
    if code == 200 and "neighbors" in res:
        print(f"[PASS] 5. GET /api/transactions/{sample_tx_id}/neighbors: Total neighbors={res['total_neighbors']}")
        passed_tests += 1
    else:
        print(f"[FAIL] 5. GET /api/transactions/{sample_tx_id}/neighbors:", code, res)

    # 6. GET /api/transactions/{tx_id}/explanation
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/transactions/{sample_tx_id}/explanation")
    if code == 200 and "explanation_text" in res:
        print(f"[PASS] 6. GET /api/transactions/{sample_tx_id}/explanation: Top features count={len(res['top_contributing_model_features'])}")
        passed_tests += 1
    else:
        print(f"[FAIL] 6. GET /api/transactions/{sample_tx_id}/explanation:", code, res)

    # 7. POST /api/predict (Test 422 for timestep 1 tx_id, test 200 for test set tx_id)
    total_tests += 1
    # 7a. Timestep 1 tx_id (230425980) -> Expect HTTP 422
    code_422, res_422 = make_request(f"{BASE_URL}/api/predict", method="POST", body={"tx_id": 230425980})

    # 7b. Test set tx_id from page 1 timestep 40/41
    page_test_res = make_request(f"{BASE_URL}/api/transactions?timestep=40&page_size=1")[1]
    predict_tx_id = page_test_res["transactions"][0]["tx_id"]
    code_200, res_200 = make_request(f"{BASE_URL}/api/predict", method="POST", body={"tx_id": predict_tx_id})

    if code_422 == 422 and code_200 == 200 and "risk_score" in res_200:
        print(f"[PASS] 7. POST /api/predict: 422 correctly handled for unpredicted tx; 200 OK for tx {predict_tx_id} -> Risk Score={res_200['risk_score']:.4f}, Level={res_200['risk_level']}")
        passed_tests += 1
    else:
        print(f"[FAIL] 7. POST /api/predict: (422 code: {code_422}), (200 code: {code_200}, res: {res_200})")

    # 8. GET /api/model/metrics
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/model/metrics")
    if code == 200 and res.get("test_pr_auc") > 0.7:
        print(f"[PASS] 8. GET /api/model/metrics: Model={res['model_name']}, PR-AUC={res['test_pr_auc']}, ROC-AUC={res['test_roc_auc']}")
        passed_tests += 1
    else:
        print("[FAIL] 8. GET /api/model/metrics:", code, res)

    # 9. GET /api/model/feature-importance
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/model/feature-importance?limit=5")
    if code == 200 and len(res.get("features", [])) == 5:
        print(f"[PASS] 9. GET /api/model/feature-importance: Top feature={res['features'][0]['feature']} (importance={res['features'][0]['importance']:.4f})")
        passed_tests += 1
    else:
        print("[FAIL] 9. GET /api/model/feature-importance:", code, res)

    # 10. GET /api/timesteps
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/timesteps")
    if code == 200 and isinstance(res, list) and len(res) == 49:
        print(f"[PASS] 10. GET /api/timesteps: Returned {len(res)} timesteps (Timestep 1 txs={res[0]['total_transactions']})")
        passed_tests += 1
    else:
        print("[FAIL] 10. GET /api/timesteps:", code, res)

    # 11. GET /api/risk-distribution
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/risk-distribution")
    if code == 200 and "counts" in res:
        print(f"[PASS] 11. GET /api/risk-distribution: Counts={res['counts']}")
        passed_tests += 1
    else:
        print("[FAIL] 11. GET /api/risk-distribution:", code, res)

    # 12. GET /api/transactions/{tx_id}/investigation (NEW PHASE 6)
    total_tests += 1
    code, res = make_request(f"{BASE_URL}/api/transactions/{predict_tx_id}/investigation")
    if code == 200 and "investigation_priority" in res and "investigation_score" in res:
        print(f"[PASS] 12. GET /api/transactions/{predict_tx_id}/investigation: Priority='{res['investigation_priority']}', Score={res['investigation_score']}")
        passed_tests += 1
    else:
        print(f"[FAIL] 12. GET /api/transactions/{predict_tx_id}/investigation:", code, res)

    print("============================================================")
    print(f"RESULTS: Passed {passed_tests} / {total_tests} tests.")
    print("============================================================")
    if passed_tests == total_tests:
        sys.exit(0)
    else:
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
