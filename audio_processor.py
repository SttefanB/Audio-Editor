import numpy as np
from scipy.io import wavfile
import scipy.signal as signal

class AudioProcessor:
    def __init__(self):
        self.fs=None
        self.data=None
        self.duration=0
        
    def incarca_fisier(self,cale):
        #Citim fisierul, normalizam datele intre -1 si 1
        try:
            self.fs, self.data=wavfile.read(cale)
            
        #In cazul unui fisier stereo pastram un singur  canal
            if len(self.data.shape)>1:
                self.data=self.data[:,0]
            
            self.data=self.data.astype(float)
            
            #Normalizare semnal in intervalul [-1,1]
            maxim=np.max(np.abs(self.data))
            if maxim>0:
                self.data=self.data/maxim
            
            self.duration=len(self.data)/self.fs
            return True
        except Exception as e:
           print(f"Eroare la incarcare :{e}")
           return False
       
    def salvare_fisier(self, cale_noua):
         #Salvam semnalul modificat
         if self.data is None: return False
         try:
             wavfile.write(cale_noua,self.fs, self.data.astype(np.float32)) #Wavfine asteapta date in format float32
             return True
         except Exception as e:
            print(f"Eroare la salvare: {e}")
            return False
    
    def aplica_filtru(self, tip_filtru, frecventa_taiere):
        '''
        Specificam tipul filtrului si frecventa de taiere
        '''
        if self.data is None: return
        
        #Calculam frecventa Nyquist
        f_nyquist = 0.5 * self.fs
        
        #Normalizam frecventa
        normal_cutoff = frecventa_taiere / f_nyquist
        
        #Luam masuri de siguranta pentru valori extreme ale frecventei
        # Măsuri de siguranță matematică
        if normal_cutoff >= 1: normal_cutoff = 0.99
        if normal_cutoff <= 0: normal_cutoff = 0.01
        
        #Calculam ponderile filtrului Butterworth
        b, a = signal.butter(5, normal_cutoff, btype=tip_filtru, analog=False)
        
        self.data = signal.filtfilt(b, a, self.data)
        
    def aplica_filtru_band_pass(self, low_cut, high_cut):
        """
        Filtru Trece Bandă (Band Pass).
        Lasă să treacă doar frecvențele dintre low_cut și high_cut.
        Util pentru efectul de 'telefon' (300Hz - 3000Hz).
        """
        if self.data is None: return
        
        nyquist = 0.5 * self.fs
        low = low_cut / nyquist
        high = high_cut / nyquist
        
        # Verificări de siguranță
        if low <= 0: low = 0.01
        if high >= 1: high = 0.99
        if low >= high: low = high - 0.1

        # btype='band' specifică Band Pass
        b, a = signal.butter(5, [low, high], btype='band')
        self.data = signal.filtfilt(b, a, self.data)
         
    def calcul_energie_cadru(self,cadru):
       return np.sum(cadru**2)
   
    def calcul_rtz_cadru(self,cadru):
        return np.sum(abs(np.sign(cadru[:-1])-np.sign(cadru[1:])))/2
    def calcul_parametrii_timp_scurt(self):
        #Aplicam functiile de mai sus utilizand o fereastra de 20ms
        if self.data is None:
            return None, None
        Tf=0.02 ##Dimensiunea ferestrei
        N=int(Tf*self.fs) #Numar esantioane pe cadru
        energie=[]
        Rtz=[]
        #Vom parcurge semnalul
        for i in range(0, len(self.data)-N, N):
            cadru=self.data[i:i+N]
        #Se calculeaza parametrii acestui cadru
            en=self.calcul_energie_cadru(cadru)
            rata_zero=self.calcul_rtz_cadru(cadru)
            energie.append(en)
            Rtz.append(rata_zero)
        return np.array(energie), np.array(Rtz)
    def calcul_autocorelatie(self):
        #Se calculeaza autocorelatia pentru o portiune din mijlocul semnalului sonor
        if self.data is None: return None
        
        mijloc=len(self.data)//2
        N=int(0.02*self.fs)
        cadru=self.data[mijloc:mijloc+N]
        
        #Aplicam formula standard
        r=np.correlate(cadru,cadru, "full")
        return r[len(r)//2:]
    def calcul_spectru_semnal(self):
        #Calculam spectrul semnalului folosind FFT
        #Vom avea pe OX frecventa semnalului si pe OY Magnitudinea semnalului
        if self.data is None:
            return None, None
        N=len(self.data)
        #Calculam FFT
        yf=np.fft.fft(self.data)
        #Generam frecventele 
        xf=np.fft.fftfreq(N,1/self.fs)
        #Vom ignora frecventele negative din spectru
        jum_N=N//2
        frecvente=xf[:jum_N]
        #Ne intereseaza modulul numarului complex
        amplitudini=np.abs(yf[:jum_N])
        return frecvente, amplitudini
    
    def get_window_spectrum(self, center_sample, win_size):
       """
       Returnează (freqs, mags) pentru o fereastră centrată în center_sample (indices).
       win_size must be power-of-two (e.g., 1024, 2048). Dacă nu, se face zero-pad.
       """
       if self.data is None or self.fs is None:
           return None, None

       half = win_size // 2
       start = int(center_sample) - half
       end = int(center_sample) + half
       # handle boundaries with zero-padding
       if start < 0 or end > len(self.data):
           frame = np.zeros(win_size, dtype=np.float64)
           s = max(0, start)
           e = min(len(self.data), end)
           insert_from = s - start
           frame[insert_from:insert_from + (e - s)] = self.data[s:e]
       else:
           frame = self.data[start:end]

       # apply window (Hann)
       win = np.hanning(win_size)
       frame = frame * win

       # FFT
       N = win_size
       yf = np.fft.rfft(frame, n=N)
       mags = np.abs(yf)
       freqs = np.fft.rfftfreq(N, d=1.0 / self.fs)
       return freqs, mags
    

        
        