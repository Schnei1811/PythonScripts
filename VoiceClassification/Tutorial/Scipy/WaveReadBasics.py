import struct
import scipy.io.wavfile as wavfile


rate, data = wavfile.read('TestBye.wav')
print(data[:20])



fin = open("TestBye.wav","rb") # Read wav file, "r flag" - read, "b flag" - binary
ChunkID=fin.read(4) # First four bytes are ChunkID which must be "RIFF" in ASCII
print("ChunkID=",ChunkID)
ChunkSizeString=fin.read(4) # Total Size of File in Bytes - 8 Bytes
ChunkSize=struct.unpack('I',ChunkSizeString) # 'I' Format is to to treat the 4 bytes as unsigned 32-bit inter
TotalSize=ChunkSize[0]+8 # The subscript is used because struct unpack returns everything as tuple
print("TotalSize=",TotalSize)
DataSize=TotalSize-44 # This is the number of bytes of Data
print("DataSize=",DataSize)
Format=fin.read(4) # "WAVE" in ASCII
print("Format=",Format)
SubChunk1ID=fin.read(4) # "fmt " in ASCII
print("SubChunk1ID=",SubChunk1ID)
SubChunk1SizeString=fin.read(4) # Should be 16 (PCM, Pulse Code Modulation)
SubChunk1Size=struct.unpack("I",SubChunk1SizeString) # 'I' format to treat as unsigned 32-bit integer
print("SubChunk1Size=",SubChunk1Size[0])
AudioFormatString=fin.read(2) # Should be 1 (PCM)
AudioFormat=struct.unpack("H",AudioFormatString) # 'H' format to treat as unsigned 16-bit integer
print("AudioFormat=",AudioFormat[0])
NumChannelsString=fin.read(2) # Should be 1 for mono, 2 for stereo
NumChannels=struct.unpack("H",NumChannelsString) # 'H' unsigned 16-bit integer
print("NumChannels=",NumChannels[0])
SampleRateString=fin.read(4) # Should be 44100 (CD sampling rate)
SampleRate=struct.unpack("I",SampleRateString)
print("SampleRate=",SampleRate[0])
ByteRateString=fin.read(4) # 44100*NumChan*2 (88200 - Mono, 176400 - Stereo)
ByteRate=struct.unpack("I",ByteRateString) # 'I' unsigned 32 bit integer
print("ByteRate=",ByteRate[0])
BlockAlignString=fin.read(2) # NumChan*2 (2 - Mono, 4 - Stereo)
BlockAlign=struct.unpack("H",BlockAlignString) # 'H' unsigned 16-bit integer
print("BlockAlign=",BlockAlign[0])
BitsPerSampleString=fin.read(2) # 16 (CD has 16-bits per sample for each channel)
BitsPerSample=struct.unpack("H",BitsPerSampleString) # 'H' unsigned 16-bit integer
print("BitsPerSample=",BitsPerSample[0])
SubChunk2ID=fin.read(4) # "Data" in ASCII
print("SubChunk2ID=",SubChunk2ID)
SubChunk2SizeString=fin.read(4) # Number of Data Bytes, Same as DataSize
SubChunk2Size=struct.unpack("I",SubChunk2SizeString)
print("SubChunk2Size=",SubChunk2Size[0])
S1String=fin.read(2) # Read first Data, number between -32768 and 32767
S1=struct.unpack("h",S1String)
print("S1=",S1[0])
S2String=fin.read(2) # Read second Data, number between -32768 and 32767
S2=struct.unpack("h",S2String)
print("S2=",S2[0])
S3String=fin.read(2) # Read third Data, number between -32768 and 32767
S3=struct.unpack("h",S3String)
print("S3=",S3[0])
S4String=fin.read(2) # Read fourth Data, number between -32768 and 32767
S4=struct.unpack("h",S4String)
print("S4=",S4[0])
S5String=fin.read(2) # Read fifth Data, number between -32768 and 32767
S5=struct.unpack("h",S5String)
print("S5=",S5[0])
fin.close()




