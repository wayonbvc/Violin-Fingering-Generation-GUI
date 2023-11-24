
import tensorflow as tf
import numpy as np


symbolic_model_path = "symbolic_model"
audio_model_path = "audio_model"
autoregressive_model_path = "autoregressive_model"

class violin_fingering_model(object):

    def inference(self, pitches, starts, durations, strings, positions, fingers, audio_features = [], mode='A'):
        
        seq_len = 32
        input_length = len(pitches)
        trans_starts = starts.copy()
        trans_durations = durations.copy()
        trans_pitches = pitches.copy()
                
        def find_nearest(array,value):
            idx,val = min(enumerate(array), key=lambda x: abs(x[1]-value))
            return val

        pitch_category = np.linspace(55, 100, 46)
        onset_category = np.linspace(-1, 4, 81)
        duration_category = np.linspace(0, 4, 65)

        for i in range(input_length):
            trans_pitches[i] = find_nearest(pitch_category,pitches[i])
            trans_starts[i] = find_nearest(onset_category,trans_starts[i])
            trans_durations[i] = find_nearest(duration_category,trans_durations[i])
            

        # Map feature value to class. 0 is left for padding token.        
        feature_dict_list = [{} for i in range(3)]
        for i in range(3):
            feature_list = [pitch_category,onset_category,duration_category]
            for j,num in enumerate(feature_list[i]):
                feature_dict_list[i][num]=j+1
                
        # Map (string, position, finger) into a number, and map the number into 240 class. 0, 1 are left for padding tokens.
        spf_dict={}
        spf_reverse_dict={}
        count=2
        for i in range(1,5):
            for j in range(1,13):
                for k in range(1,6):
                    idx = i*1000+j*10+k*1
                    spf_dict[idx]=count
                    spf_reverse_dict[count]=idx
                    count+=1
        
        # From class to string, position, finger
        def get_spf(x):
            idx = spf_reverse_dict[x]
            s_list= ["G","D","A","E"]
            s = s_list[int(idx/1000)-1]
            p = int((idx%1000)/10)
            f = idx%10 - 1
            return [s,p,f]
                
        # Features are converted to categorical inputs
        
        for i in range(input_length):
            trans_pitches[i] = feature_dict_list[0][trans_pitches[i]]
            trans_starts[i] = feature_dict_list[1][trans_starts[i]]
            trans_durations[i] = feature_dict_list[2][trans_durations[i]]
            

        
        if mode=="Sym_Audio" or mode=="Symbolic":              
            pad_length = seq_len - input_length%seq_len
            for i in range(pad_length):
                trans_pitches = np.append(trans_pitches,[0])
                trans_starts = np.append(trans_starts,[0])
                trans_durations = np.append(trans_durations,[0])
                
            trans_pitches = trans_pitches.reshape((int(len(trans_pitches)/seq_len),seq_len))
            trans_starts = trans_starts.reshape((int(len(trans_starts)/seq_len),seq_len))
            trans_durations = trans_pitches.reshape((int(len(trans_durations)/seq_len),seq_len))           
        
        
            if mode=="Sym_Audio":
                print("Using Audio Model")
                pad_length = seq_len - input_length%seq_len
                trans_audio_features = np.concatenate((audio_features, np.zeros((pad_length,audio_features.shape[1]))))
                trans_audio_features = trans_audio_features.reshape((int(len(trans_audio_features)/seq_len),seq_len,audio_features.shape[1]))
                audio_model = tf.keras.models.load_model(audio_model_path)
                prediction = audio_model.predict(x=[[trans_pitches,trans_starts,trans_durations],[trans_audio_features]])
                
            elif mode=="Symbolic":
                print("Using Symbolic Model")
                symbolic_model = tf.keras.models.load_model(symbolic_model_path)
                prediction = symbolic_model.predict(x=[trans_pitches,trans_starts,trans_durations])
                
            prediction[:,:,:2] = 0
            arg_prediction = np.argsort(-prediction,axis=-1)  
            string_predictions = []
            position_predictions = []
            finger_predictions = []
            
            predict_fingers = []
            for i in range(arg_prediction.shape[0]):
                for j in range(arg_prediction.shape[1]):
                    s,p,f = get_spf(arg_prediction[i][j][0])
                    string_predictions.append(s)
                    position_predictions.append(p)
                    finger_predictions.append(f)
                    predict_fingers.append(arg_prediction[i][j][0])
                
        elif mode=="Autoregressive":
            
            context_pitches = trans_pitches.copy()
            context_starts = trans_starts.copy()
            context_durations = trans_durations.copy()    
            context_seq_len = 16
            half_con_seq_len = int(context_seq_len/2)
            context_pitches = np.concatenate( (np.zeros(half_con_seq_len),context_pitches,np.zeros(half_con_seq_len)))
            context_starts = np.concatenate( (np.zeros(half_con_seq_len),context_starts,np.zeros(half_con_seq_len)))
            context_durations  = np.concatenate ((np.zeros(half_con_seq_len),context_durations,np.zeros(half_con_seq_len)))
    
            
            context_p=[]
            context_s=[]
            context_d=[]
            
            
            for i in range(input_length):
                context_p.append(context_pitches[i:i+context_seq_len+1])
                context_s.append(context_starts[i:i+context_seq_len+1])
                context_d.append(context_durations[i:i+context_seq_len+1])
         
            context_p = np.array(context_p)
            context_s = np.array(context_s)
            context_d = np.array(context_d)
            context_fingers = np.zeros(len(context_pitches))
                     
            context_model = tf.keras.models.load_model(autoregressive_model_path)
            string_predictions = []
            position_predictions = []
            finger_predictions = []
            for i in range(input_length):
                context_f=[]
                tmp = context_fingers[i:i+context_seq_len+1].copy()
                tmp[half_con_seq_len] = 1
                tmp[half_con_seq_len+1:] = 0
                context_f.append(tmp)    
                context_f = np.array(context_f)
                
                prediction = context_model.predict(x=[context_p[i:i+1],context_s[i:i+1],context_d[i:i+1],context_f],verbose = 0)
                prediction[:,:2] = 0
                arg_prediction = np.argsort(-prediction,axis=-1)  
                s,p,f = get_spf(arg_prediction[0][0])
                context_fingers[i+half_con_seq_len] = arg_prediction[0][0]
                string_predictions.append(s)
                position_predictions.append(p)
                finger_predictions.append(f)

                print(pitches[i],s,p,f)
            
       
            
        return string_predictions[:input_length], position_predictions[:input_length], finger_predictions[:input_length]





