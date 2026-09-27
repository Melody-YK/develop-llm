import urllib.request

targets = [
    "https://pypi.tuna.tsinghua.edu.cn/simple/pip/",
    "https://mirrors.aliyun.com/pypi/simple/pip/",
    "https://pypi.mirrors.ustc.edu.cn/simple/pip/",
    "https://mirrors.cloud.tencent.com/pypi/simple/pip/",
]
for m in targets:
    try:
        req = urllib.request.Request(m, headers={"User-Agent": "pip/24.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read(200)
            print(r.status, m, "| bytes:", len(body))
    except Exception as e:
        print("FAIL", m, "|", repr(e)[:120])
