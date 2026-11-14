import renderer
import chorale_engine

n = 1298074214633706907132624082305023

chorale_engine.write_musicxml(215, chorale_engine.encode(215, n), "score.musicxml")
with open("score.musicxml", encoding="utf-8") as file:
    music_xml = file.read()
renderer.writeScore(music_xml, "test.png", "png", 2000, 1.0)