# Violin-Fingering-Generation-GUI
We provide a deep-learning-based violin fingering generation system that directly exhibits the generated results on the GUI. Given a violin score, our system can generate violin fingering with three models: symbolic, symbolic-audio, and autoregressive. The symbolic model takes the violin score as input and generates a sequence of fingerings at a time. The symbolic-audio model takes both the violin score and the violin audio as inputs, generating fingering with the guidance of the audio. The autoregressive model generates one fingering at a time, taking the violin score and the previously generated fingering as inputs. Users can save the generated fingering as an XML file and open and edit it with any music notation software.

# Install
1. Clone the repository and install the package in the **requirements.txt** (with python 3.11)   
    `pip install -r requirements.txt`
2. To use the symbolic-audio model, please install the package **fluidsynth**. (https://github.com/FluidSynth/fluidsynth)

# Usage

1. Run the file **violin_app.py**
2. Load an XML file that contains a violin score.
3. Select a generation model (symbolic, symbolic-audio, autoregressive). To use the symbolic-audio model, users have to import an audio file.
4. Click **recommend** to generate violin fingering. The generated fingering will be shown on the GUI.
5. If users are unsatisfied with the output, press **clear fingering** and try another model.
6. Press **Save xml** to save the score with generated fingerings. Users can open and edit it with any music notation software.
