import midi
import matplotlib.pyplot as plt
import numpy as np
import argparse

# parser = argparse.ArgumentParser(description="Turn a .midi file into a graph")
# parser.add_argument('midi_file', help='Path to .midi file to process')
#
# args = parser.parse_args()

midi_file = 'G:Midi/Piano/deb_clai.mid'

song = midi.read_midifile(midi_file)


song.make_ticks_abs()
tracks = []


songmatrix = np.zeros((1,88))


for track in song:
    notes = [note for note in track if note.name == 'Note On']
    pitch = [note.pitch for note in notes]
    tick = [note.tick for note in notes]
    if len(tick) > 0:
        if max(tick) > maxtick:
            maxtick = max(tick)
    tracks += [tick, pitch]


for i in range(maxtick):








print(maxtick)

plt.plot(*tracks)
plt.show()
