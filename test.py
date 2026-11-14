import renderer
import chorale_engine

with open("durhack_word_list.csv", "r") as file:
    words = file.read().splitlines()

text_input = input("Enter word: ")

index = ""
for x in text_input.split():
    index += str(words.index(x)+10000)
print(index)
chorale_engine.write_musicxml(215, chorale_engine.encode(215, int(index)), "score.musicxml")
with open("score.musicxml", encoding="utf-8") as file:
    music_xml = file.read()
renderer.writeScore(music_xml, "test.png", "png", 2000, 1.0)

a, b = chorale_engine.decode_musicxml("score.musicxml")
print(b)
b = str(b)
for i in range(int(len(b)/5)):
    print(words[int(b[i*5:(i+1)*5])-10000])