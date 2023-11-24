import csv
import enum
import logging
import subprocess
import tempfile
from copy import deepcopy
from pathlib import Path

import pretty_midi
from library.read_csv import read_csv
from lxml import etree
import music21
from violin_fingering_model import violin_fingering_model
from library import generate_audio_feature

NOT_APPLICABLE = "N/A"


class PreferenceMode(enum.Enum):
    Symbolic = "Symbolic"
    Sym_Audio = "Sym_Audio"
    Autoregressive = "Autoregressive"
    #LOWEST_MODE = "lowest"
    #NEAREST_MODE = "nearest"


class Strings(enum.IntEnum):
    N = 0
    G = 1
    D = 2
    A = 3
    E = 4


position_tamplate = """<direction placement="above">
            <direction-type>
                <words relative-y="20.00">{position}</words>
            </direction-type>
        </direction>"""

string_tamplate = """<direction placement="above">
        <direction-type>
          <words relative-x="-1.81" relative-y="45.12">{string}</words>
          </direction-type>
        </direction>"""

notations_sample = """<notations>
            <technical>
                <fingering default-x="2" default-y="31" placement="above">{finger}</fingering>
                <!--down-bow default-x="0" default-y="10" placement="above"/-->
            </technical>
        </notations>"""


class _Parser:
    def __init__(self, musescore_path):
        self.musescore_path = musescore_path
        self.audio_features = []
    @property
    def output_xmltree(self):
        if not hasattr(self, "_output_xmltree"):
            self.reset_output_xmltree()
        return self._output_xmltree

    @output_xmltree.setter
    def output_xmltree(self, output_xmltree):
        self._output_xmltree = output_xmltree

    @property
    def output_notes(self):
        if not hasattr(self, "_output_notes"):
            self.reset_output_notes()
        return self._output_notes

    @output_notes.setter
    def output_notes(self, output_notes):
        self._output_notes = output_notes

    @property
    def score(self):
        if not hasattr(self, "_score"):
            raise RuntimeError("No score property, please load_musicxml first")
        return self._score

    @score.setter
    def score(self, score):
        self._score = score

    @property
    def import_notes(self):
        if not hasattr(self, "_import_notes"):
            raise RuntimeError("No import notes, please import csv first")
        return self._import_notes

    @import_notes.setter
    def import_notes(self, import_notes):
        self._import_notes = import_notes

    @staticmethod
    def write_dict_csv(notes, csv_path):
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(notes[0].keys()))
            writer.writeheader()
            for note in notes:
                writer.writerow(note)

    @staticmethod
    def read_dict_csv(csv_path):
        notes_list = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=",")
            for row in reader:
                notes_list.append(row)
        return notes_list

    def _update_output_notes(self, notes):
        for index, note in enumerate(notes):
            # those property should be overwrite
            for note_property in ["string", "position", "finger"]:
                self.output_notes[index][note_property] = note[note_property]

    def _update_output_xmltree(self, notes):
        output_xmlnotes = self.output_xmltree.getroot().findall(".//note")
        previous_string = None

        output_measures = self.output_xmltree.getroot().findall(".//measure")
        
        index_to_xml_index = {}
        count = 0
        xml_index = 0
        for measure in output_measures:
            voice_onsets = [0,0,0,0]
            xml_index_list = []
            onset_list = []
            pitch_list = []
            measure_notes = measure.findall(".//note")
            for note in measure_notes:
                try:
                    if note.find("notehead").text=="diamond":
                        xml_index += 1
                        continue
                except:
                    pass
                try:
                    onset = voice_onsets[int(note.find("voice").text)-1]
                    duration = int(note.find("duration").text)
                    voice_onsets[int(note.find("voice").text)-1]+=duration
                    
                except:
                    onset = voice_onsets[0]
                try:
                    tied = note.find("notations").find("tied")
                    if tied.attrib["type"]=="stop":
                        xml_index += 1
                        continue
                except:
                    pass
                if note.find("pitch"):      
                    step = note.find("pitch").find("step").text
                    octave = int(note.find("pitch").find("octave").text)
                    try:alter= int(note.find("pitch").find("alter").text)
                    except: alter=0
                    pitch = int(music21.pitch.Pitch(step=step,  octave=octave).ps)+alter
                    pitch_list.append(pitch)
                    xml_index_list.append(xml_index)
                    onset_list.append(onset)
                xml_index += 1
            result_list = [i for _, _,i in sorted(zip(onset_list,pitch_list, xml_index_list))]
            for num in result_list:
                index_to_xml_index[count] = num
                count+=1
        


        for index, note in enumerate(notes):

            pred_finger = note["finger"]
            if pred_finger != NOT_APPLICABLE:
                finger_element = etree.XML(notations_sample.format(finger=pred_finger))
                output_xmlnotes[index_to_xml_index[index]].append(finger_element)

            pred_position = note["position"]
            UPDATE_POSITION = False
            if UPDATE_POSITION and pred_position != NOT_APPLICABLE:
                position_element = etree.XML(
                    position_tamplate.format(position=pred_position)
                )
                output_xmlnotes[index_to_xml_index[index]].addprevious(position_element)

            # Only show string when it changes.
            pred_string = (
                Strings[note["string"]].name
                if note["string"] != NOT_APPLICABLE
                else note["string"]
            )
            if pred_string != NOT_APPLICABLE and pred_string != previous_string:
                string_element = etree.XML(string_tamplate.format(string=pred_string))
                output_xmlnotes[index_to_xml_index[index]].addprevious(string_element)
                previous_string = pred_string

    def _verify_import_notes(self, import_notes):

        assert len(import_notes) == len(
            self.output_notes
        ), "the total number of notes from import csv should be same as original xml"
        for index, note in enumerate(import_notes):
            # those property should not be change
            for note_property in [
                "pitch",
                "time_start",
                "duration",
            ]:
                assert (
                    float(self.output_notes[index][note_property]) == float(note[note_property])
                ), f"{self.output_notes[index][note_property]} != {note[note_property]} note property:{note_property} from import csv should be same as original xml"

    @staticmethod
    def _convert_beat_type_naming(beat_type):
        
        beat_type_naming_lookup = {
            "eighth": "8th",
            "quarter": "16th",
            "32nd": "32th",
        }
        return beat_type_naming_lookup.get(beat_type, beat_type)

    def _get_music_property(self, index, note):

        if note.beat>0:
            beat = note.beat
        else:
            beat = 1

        music_property = {
            "pitch": note.pitch.midi,
            "measure": note.measureNumber,
            "polyphonic":0,
            "time_start": float(beat) - 1 ,
            "duration": note.duration.quarterLength,
            "beat_type": self._convert_beat_type_naming(note.duration.type),
            "string": NOT_APPLICABLE,
            "position": NOT_APPLICABLE,
            "finger": NOT_APPLICABLE,
        }
        # make all value string
        music_property = {key: str(value) for key, value in music_property.items()}
        return music_property

    def _get_notes_from_source(self, input_xmltree):

        input_measures = input_xmltree.getroot().findall(".//measure")
        note_list = []
        
        for i,measure in enumerate(input_measures):
            tmp_list=[]
            voice_onsets = [0,0,0,0]
            measure_notes = measure.findall(".//note")
            for note in measure_notes:
                try:
                    if note.find("notehead").text=="diamond":
                        continue
                except:
                    pass
                
                try:
                    onset = voice_onsets[int(note.find("voice").text)-1]
                    duration = int(note.find("duration").text)/self.divisions
                    beat_type = note.find("type").text
                    voice_onsets[int(note.find("voice").text)-1]+=duration

                except:
                    onset = voice_onsets[0]
                
                if note.find("pitch"): 
                    step = note.find("pitch").find("step").text
                    octave = int(note.find("pitch").find("octave").text)
                    try:alter= int(note.find("pitch").find("alter").text)
                    except: alter=0
                    pitch = int(music21.pitch.Pitch(step=step,  octave=octave).ps)+alter
                    try:
                        tied = note.find("notations").find("tied")
                        if tied.attrib["type"]=="stop":
                            try:
                                tmp_list[-1]["duration"] = str(float(tmp_list[-1]["duration"])+duration)
                            except:
                                note_list[-1]["duration"] = str(float(note_list[-1]["duration"])+duration)

                            continue
                    except:
                        pass
                    music_property = {
                        "pitch": pitch,
                        "measure": i+1,
                        "polyphonic":0,
                        "time_start": onset,
                        "duration": duration,
                        "string": NOT_APPLICABLE,
                        "position": NOT_APPLICABLE,
                        "finger": NOT_APPLICABLE,
                    }
                    music_property = {key: str(value) for key, value in music_property.items()}
                    tmp_list.append(music_property)
            tmp_list = sorted(tmp_list, key=lambda k: (k["time_start"],k["pitch"]))
            j = 1
  
            while j < len(tmp_list):
                if tmp_list[j]["measure"]==tmp_list[j-1]["measure"] and tmp_list[j]["time_start"]==tmp_list[j-1]["time_start"]:
                    tmp_list[j]["polyphonic"] = 1
                j+=1
                
            note_list = note_list + tmp_list

        return note_list

    @staticmethod
    def _convert_beat_type(beat_type):
        beat_type_lookup = {
            "": 0,
            "whole": 1,
            "half": 2,
            "4th": 3,
            "8th": 4,
            "16th": 5,
            "32th": 6,
        }
        if beat_type in beat_type_lookup.keys():
            for name, index in beat_type_lookup.items():
                if name == beat_type:
                    return index
            else:
                raise LookupError("Unknown beat type")
        elif beat_type in beat_type_lookup.values():
            for name, index in beat_type_lookup.items():
                if index == beat_type:
                    return name
            else:
                raise LookupError("Unknown beat type")
        else:
            raise ValueError("Cannot convert beat type")

    def _convert_list_to_notes(
        self,
        pitches,
        starts,
        durations,
        pred_string,
        pred_position,
        pred_finger,
    ):
        notes_list = []
        n_notes = len(pitches)
        for i in range(n_notes):
            music_property = {
                "pitch": pitches[i],
                "time_start": starts[i],
                "duration": durations[i],
                "string": pred_string[i],
                "position": pred_position[i],
                "finger": pred_finger[i],
            }
            music_property = {key: str(value) for key, value in music_property.items()}
            notes_list.append(music_property)
        self._verify_import_notes(notes_list)
        return notes_list

    @staticmethod
    def _show_result_by_pretty_midi(pitches, pred_string, pred_position, pred_finger):
        print(
            "pitch".ljust(9),
            "".join(
                [pretty_midi.note_number_to_name(number).rjust(4) for number in pitches]
            ),
        )
        print(
            "string".ljust(9),
            "".join([s.rjust(4) for s in pred_string]),
        )
        print(
            "position".ljust(9),
            "".join([p.rjust(4) for p in pred_position]),
        )
        print(
            "finger".ljust(9),
            "".join([f.rjust(4) for f in pred_finger]),
        )


#
# Public API
#
class Parser(_Parser):
    def is_musicxml_loaded(self):
        return hasattr(self, "xmltree")

    def is_import_loaded(self):
        return hasattr(self, "_import_notes")

    def save_musicxml(self, xml_path):
        logging.debug("saving musicxml to %s", xml_path)

        with open(xml_path, "wb") as f:
            self.output_xmltree.write(
                f,
                xml_declaration=True,
                encoding=self.output_xmltree.docinfo.encoding,
                standalone=self.output_xmltree.docinfo.standalone,
                method="xml",
            )
        logging.debug("XML created successfully")

    def save_as_pdf(self, pdf_path):
        logging.debug("saving pdf to %s", pdf_path)
        # Save temp musicxml and convert to pdf
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_xml = Path(tmpdir, "temp_for_convert_to_pdf").with_suffix(".xml")
            self.save_musicxml(temp_xml)
            cmd_convert_pdf = list(
                map(str, [self.musescore_path, "-o", pdf_path, temp_xml])
            )
            subprocess.run(cmd_convert_pdf, shell=True, text=True, check=True)
        logging.debug("PDF created successfully")

    def save_as_csv(self, csv_path):
        logging.debug("saving CSV to %s", csv_path)
        self.write_dict_csv(self.output_notes, csv_path)
        logging.debug("CSV created successfully")

    def load_musicxml(self, filepath):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} is not exists")

        if filepath.suffix not in [".xml", ".musicxml"]:
            raise TypeError(f"{filepath.suffix} is not supported")

        xmlparser = etree.XMLParser(dtd_validation=False)
        try:
            self.xmltree = etree.parse(str(filepath), xmlparser)
            self.xmlnotes = self.xmltree.getroot().findall(".//note")
            self.divisions = float(
                self.xmltree.find("./part/measure/attributes/divisions").text
            )
        except Exception:
            logging.error("%s is not a valid XML or can not parse", filepath.name)
            raise

        try:
            #self.score = converter.parse(filepath, format="musicxml").flat
            self.source_notes = self._get_notes_from_source(self.xmltree)
        except:
            logging.error("converting musicxml to notes fail")
            raise

        logging.debug("load musicxml successfully")
        self.reset_output_notes()
        self.temp_notes_path = filepath.parent / "temp_notes.csv"
        self.save_as_csv(self.temp_notes_path)        
        return self.output_notes

    def reset_output_notes(self):
        self.output_notes = deepcopy(self.source_notes)

    def reset_output_xmltree(self):
        self.output_xmltree = deepcopy(self.xmltree)

    def load_import(self, filepath):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} is not exists")

        if filepath.suffix not in [".csv"]:
            raise TypeError(f"{filepath.suffix} is not supported")

        try:
            self.import_notes = self.read_dict_csv(filepath)
            self._verify_import_notes(self.import_notes)
        except Exception:
            logging.error("Something wrong happened during import %s", filepath.name)
            raise

        self.last_import = filepath
        logging.debug("get notes from import successfully")
        self.show_csv()
        return self.import_notes
    
    def load_import_audio(self, filepath):
        filepath = Path(filepath).resolve()
        if not filepath.exists():
            raise FileNotFoundError(f"{filepath} is not exists")

        if filepath.suffix not in [".mp3",".wav",".m4a"]:
            raise TypeError(f"{filepath.suffix} is not supported")

        try:
            self.audio_features = generate_audio_feature.generate_features(filepath, self.source_notes)
            #self._verify_import_notes(self.import_notes)
        except Exception:
            logging.error("Something wrong happened during import %s", filepath.name)
            raise

        #self.last_import = filepath
        logging.debug("get audio from import successfully")
        #self.show_csv()

    def show_csv(self):
        self._update_output_notes(self.import_notes)
        self._update_output_xmltree(self.output_notes)

    def recommend(self, mode=PreferenceMode.Symbolic):
        assert isinstance(mode, PreferenceMode)
        input = read_csv(self.temp_notes_path)
        #beat_types,
        pitches, starts, durations,  strings, positions, fingers = (
            input["pitches"],
            input["starts"],
            input["durations"],
            #input["beat_types"],
            input["strings"],
            input["positions"],
            input["fingers"],
        )
        

        model = violin_fingering_model()
        pred_str, pred_pos, pred_fin = model.inference(
            pitches=pitches,
            starts=starts,
            durations=durations,
            #beat_types=beat_types,
            strings=strings,
            positions=positions,
            fingers=fingers,
            audio_features = self.audio_features,
            mode=mode.value,
        )

        # convert_ndarray_to_list

        pred_string = [str(s) for s in pred_str]
        pred_position = [str(p) for p in pred_pos]
        pred_finger = [str(f) for f in pred_fin]
        #self._show_result_by_pretty_midi(pitches, pred_string, pred_position, pred_finger)

        recommend_notes = self._convert_list_to_notes(
            pitches,
            starts,
            durations,
            pred_string,
            pred_position,
            pred_finger,
        )
        logging.debug("get notes from recommend successfully")
        self._update_output_notes(recommend_notes)
        self._update_output_xmltree(self.output_notes)

    def get_preview_images(self):
        # Save temp musicxml and Convert to image png for preview
        temp_xml = Path(tempfile.mkdtemp(), "temp_for_preview").with_suffix(".xml")
        self.save_musicxml(temp_xml)

        output_png = temp_xml.with_name(f"{temp_xml.stem}.png")
        cmd_convert_png = list(
            map(str, [self.musescore_path, "-o", output_png, temp_xml])
        )
        subprocess.run(cmd_convert_png, shell=True, text=True, check=True)
        preview_images = list(output_png.parent.glob(f"{output_png.stem}-*.png"))
        return preview_images
