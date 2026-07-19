import requests

def test_server_alive():
    url = "http://127.0.0.1:8000/test_api"
    resp = requests.get(url, timeout=5)
    # 只校验状态码200，证明服务正常
    assert resp.status_code == 200

def test_predict_api():
    url = "http://127.0.0.1:8000/predict"
    resp = requests.post(url, None, timeout=5)
    assert resp.status_code == 422, "预测接口未返回预期的422错误，可能服务未启动或接口异常"
