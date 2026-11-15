"""The web app: payload counter, random chorales, checksum repair, and old scores still decoding."""

import io
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

pytest.importorskip("flask")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "webapp"))
import app as web  # noqa: E402

MESSAGE = "apple banana cherry"


@pytest.fixture(scope="module")
def client():
    web.chorale_pool()
    return web.app.test_client()


def encode(client, style="original"):
    page = client.post("/encode", data={"secret_message": MESSAGE, "encoding_style": style}).get_data(as_text=True)
    name = re.search(r'/download/([\w-]+\.musicxml)', page).group(1)
    return page, client.get(f"/download/{name}").data


def decode(client, data):
    return client.post("/decode", data={"encoded_file": (io.BytesIO(data), "score.musicxml")},
                       content_type="multipart/form-data").get_data(as_text=True)


def test_payload_counter(client):
    assert client.get("/payload").status_code == 200
    d = client.get("/payload", query_string={"message": MESSAGE}).get_json()
    assert d["error"] is None and d["words"] == 3 and d["chorales"] > 0
    assert d["bits"] < 50 and d["check_bits"] == (16 if web.checked() else 0)
    assert client.get("/payload", query_string={"message": "notaword"}).get_json()["error"]


def test_error_correction_checkbox_controls_word_encoding(monkeypatch):
    monkeypatch.setattr(web, "pick_chorale", lambda n, style: 10)
    encoded = []
    monkeypatch.setattr(
        web, "engine_encode",
        lambda cid, n, style, **kwargs: encoded.append((n, kwargs)) or [],
    )

    with web.app.test_request_context("/encode", method="POST", data={"use_checksum": "0"}):
        web.encode_indices([0])
    assert encoded[-1] == (10000, {})

    with web.app.test_request_context("/encode", method="POST", data={"use_checksum": "1"}):
        web.encode_indices([0])
    assert encoded[-1][1] == {"checksum": True}


@pytest.mark.parametrize("style", ["original", "fancy"])
def test_round_trip_names_the_chorale(client, style):
    page, data = encode(client, style)
    name = re.search(r"Hidden in (Chorale \d+)", page).group(1)
    out = decode(client, data)
    assert MESSAGE in out and f"Found in {name}" in out and "No damage found" in out


def test_misread_chord_is_repaired(client):
    if not web.checked():
        pytest.skip("checksum is off")
    _page, data = encode(client)
    root = ET.fromstring(data)
    bass = [n for n in root.iter("note") if n.find("voice").text == "4" and n.find("pitch") is not None]
    step = bass[len(bass) // 2].find("pitch/step")
    step.text = "CDEFGAB"[("CDEFGAB".index(step.text) + 1) % 7]
    out = decode(client, ET.tostring(root))
    assert MESSAGE in out and "The message is intact" in out


def test_old_plain_scores_still_decode(client, monkeypatch):
    monkeypatch.setattr(web, "CHECKSUM", False)
    _page, data = encode(client)
    monkeypatch.setattr(web, "CHECKSUM", True)
    assert MESSAGE in decode(client, data)


def test_ascii_parts_carry_the_checksum_and_are_repaired(client):
    if not web.checked():
        pytest.skip("checksum is off")
    from chorale_engine.checksum import with_check

    style = "original"
    length = web.ascii_chunk_length(style)
    cap = max(web.chorale_pool()[style].values())
    assert with_check(web.pack_ascii_chunk("~" * length)) < cap <= with_check(web.pack_ascii_chunk("~" * (length + 1)))
    message = "Hello, Bach! 123"
    page = client.post("/encode", data={"secret_message": message, "encoding_style": style,
                                        "encoding_method": "ascii"}).get_data(as_text=True)
    name = re.search(r'/download/([\w-]+\.zip)', page).group(1)
    data = client.get(f"/download/{name}").data
    # misread one bass note in the first part, then decode the ZIP
    import zipfile
    src = zipfile.ZipFile(io.BytesIO(data))
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w") as dst:
        for info in src.infolist():
            xml = src.read(info)
            if info.filename == "part-0001.musicxml":
                root = ET.fromstring(xml)
                bass = [n for n in root.iter("note") if n.find("voice").text == "4" and n.find("pitch") is not None]
                step = bass[len(bass) // 2].find("pitch/step")
                step.text = "CDEFGAB"[("CDEFGAB".index(step.text) + 1) % 7]
                xml = ET.tostring(root)
            dst.writestr(info.filename, xml)
    decoded = client.post("/decode", data={"encoded_file": (io.BytesIO(out.getvalue()), "parts.zip")},
                          content_type="multipart/form-data").get_data(as_text=True)
    import json
    assert json.dumps(message) in decoded  # the page hands the message to its typing effect as JSON
    assert "Repaired 1 misread chord" in decoded and "The message is intact" in decoded
