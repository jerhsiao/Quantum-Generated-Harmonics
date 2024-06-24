Project Title: Quantum Generated Harmonics

For a detailed overview with plots and results, go here: https://docs.google.com/presentation/d/1iXvqjdD4Uv0B2xInZNIjcA0EJqg3CdBuRZVyMgPhDTs/edit?usp=sharing

The objective and purpose of my system is to create a model that takes in a wav file of sound and extends it, generating harmonic, ambient music based on the frequencies of the original file. I was originally inspired by ambient music like Brian Eno. In order to incorporate some ambient sounds, I used quantum circuits to decide the harmonic frequencies, small phase and frequency shifts in the original audio, and applying delay randomly.  The purpose of my project is to innovate and explore quantum computing as a tool for sound manipulation and music creation.
When modifying the original audio data, I use quantum circuits to apply phase and frequency shifts to the audio samples based on quantum measurements. I also apply quantum randomness and complexity. 
The input audio is represented as a sequence of samples at a specific rate. Then, the key and prominent pitches are detected using the librosa library. The audio samples are encoded into quantum states using quantum circuits. Each sample influences the parameters of quantum gates, creating a unique transformation. The project processes the audio in chunks, applying quantum circuits to each chunk, accumulating results, and blending them together. 
I also apply additional audio effects like delay using quantum decisions. Finally, I generate new audio samples based on pitch mapping using more quantum circuits and the pitch mapping that I generated.

Overall Information Flow: Input audio file is read and processed using scipy and librosa. 
The processed audio data is then passed to Qiskit, where quantum circuits are constructed and simulated. Quantum measurements are then taken to modify and add to the audio samples. Then, the modified audio data is normalized and written to an output file. I also use extra functions for low pass filter and reverb to make the modified audio data sound smoother. 

The inherent randomness of quantum computers can produce creative and unique sounds. I do believe that implementing this classically would produce similar results; this project just introduces more randomness and new avenues for innovation. I think that classically, it would be easier to implement and I would have more control over the output, but through this quantum system, we get more variability and more opportunities for creativity and uniqueness. Sometimes, my system would produce less than optimal sounds when testing different samples, but it had a pretty good success rate, and I think it made some pretty unique, ambient sounds. 

