#include <iostream>
#include <vector>
#include <fstream>
#include <cmath>
#include <algorithm>
#include <complex>
#define M_PI 3.14159265358979323846

struct WAVHeader {
    char riff[4];           
    uint32_t fileSize;      
    char wave[4];           
    char fmt[4];            
    uint32_t fmtSize;       
    uint16_t audioFormat;   
    uint16_t numChannels;   
    uint32_t sampleRate;    
    uint32_t byteRate;      
    uint16_t blockAlign;    
    uint16_t bitsPerSample; 
};

class FastAudioProcessor {
private:
    std::vector<double> audioData;
    const int windowSize = 2048;
    const int hopSize = 512;
    uint32_t sampleRate;
    std::vector<double> hanningWindow;

    struct NoiseThresholds {
        const double RMS_THRESHOLD = 0.1;
        const double ZCR_THRESHOLD = 0.3;
        const double CENTROID_THRESHOLD = 0.6;
    } thresholds;

    void initializeHanningWindow() {
        hanningWindow.resize(windowSize);
        #pragma omp parallel for
        for (int i = 0; i < windowSize; i++) {
            hanningWindow[i] = 0.5 * (1 - cos(2 * M_PI * i / (windowSize - 1)));
        }
    }

    void fft(std::vector<std::complex<double>>& x) {
        const size_t N = x.size();
        if (N <= 1) return;

        // Bit-reverse copy
        size_t j = 0;
        for (size_t i = 0; i < N - 1; i++) {
            if (i < j) std::swap(x[i], x[j]);
            size_t k = N >> 1;
            while (k <= j) { j -= k; k >>= 1; }
            j += k;
        }

        // Compute FFT
        for (size_t len = 2; len <= N; len <<= 1) {
            double angle = -2 * M_PI / len;
            std::complex<double> wlen(cos(angle), sin(angle));
            for (size_t i = 0; i < N; i += len) {
                std::complex<double> w(1);
                for (size_t j = 0; j < len/2; j++) {
                    std::complex<double> u = x[i+j];
                    std::complex<double> t = w * x[i+j+len/2];
                    x[i+j] = u + t;
                    x[i+j+len/2] = u - t;
                    w *= wlen;
                }
            }
        }
    }

public:
    FastAudioProcessor() {
        initializeHanningWindow();
    }

    std::vector<double> loadWavFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open WAV file");
        }

        WAVHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

        // Validate header
        if (std::string(header.riff, 4) != "RIFF" || 
            std::string(header.wave, 4) != "WAVE" ||
            std::string(header.fmt, 4) != "fmt ") {
            throw std::runtime_error("Invalid WAV file format");
        }
        
        // Find data chunk
        char chunkID[4];
        uint32_t chunkSize;
        while (true) {
            file.read(chunkID, 4);
            file.read(reinterpret_cast<char*>(&chunkSize), 4);
            if (std::string(chunkID, 4) == "data") break;
            file.seekg(chunkSize, std::ios::cur);
        }

        // Read all data at once
        std::vector<int16_t> rawData(chunkSize / 2);
        file.read(reinterpret_cast<char*>(rawData.data()), chunkSize);

        // Convert to normalized samples and mix channels
        audioData.resize(rawData.size() / header.numChannels);
        for (size_t i = 0; i < audioData.size(); i++) {
            double sample = 0;
            for (int ch = 0; ch < header.numChannels; ch++) {
                sample += rawData[i * header.numChannels + ch];
            }
            audioData[i] = sample / (header.numChannels * 32768.0);
        }

        sampleRate = header.sampleRate;
        return audioData;
    }

    std::vector<double> extractFeatures(const std::vector<double>& windowData) {
        std::vector<double> features(3);
        
        // Apply Hanning window and compute RMS
        double sumSquares = 0;
        std::vector<double> window(windowSize);
        for (size_t i = 0; i < windowSize; i++) {
            window[i] = windowData[i] * hanningWindow[i];
            sumSquares += window[i] * window[i];
        }
        features[0] = sqrt(sumSquares / windowSize);

        // Zero Crossing Rate
        int zcr = 0;
        double prevSample = window[0];
        for (size_t i = 1; i < windowSize; i++) {
            if ((window[i] >= 0 && prevSample < 0) || 
                (window[i] < 0 && prevSample >= 0)) {
                zcr++;
            }
            prevSample = window[i];
        }
        features[1] = static_cast<double>(zcr) / windowSize;

        // Spectral Centroid
        std::vector<std::complex<double>> fftData(windowSize);
        for (size_t i = 0; i < windowSize; i++) {
            fftData[i] = std::complex<double>(window[i], 0);
        }
        
        fft(fftData);
        
        double weightedSum = 0.0;
        double sum = 0.0;
        const size_t halfWindow = windowSize / 2;
        for (size_t i = 0; i < halfWindow; i++) {
            double magnitude = std::abs(fftData[i]);
            weightedSum += i * magnitude;
            sum += magnitude;
        }
        features[2] = sum != 0 ? weightedSum / sum : 0;

        return features;
    }

    std::vector<std::pair<std::vector<double>, bool>> analyzeAudio() {
        std::vector<std::pair<std::vector<double>, bool>> results;
        const size_t numWindows = (audioData.size() - windowSize) / hopSize + 1;
        results.reserve(numWindows);

        std::vector<double> window(windowSize);
        for (size_t i = 0; i + windowSize <= audioData.size(); i += hopSize) {
            // Copy window data
            std::copy(audioData.begin() + i, audioData.begin() + i + windowSize, window.begin());
            
            // Extract features
            auto features = extractFeatures(window);
            
            // Quick noise check
            int noiseIndicators = 0;
            if (features[0] > thresholds.RMS_THRESHOLD) noiseIndicators++;
            if (features[1] > thresholds.ZCR_THRESHOLD) noiseIndicators++;
            if (features[2] / (windowSize / 2.0) > thresholds.CENTROID_THRESHOLD) noiseIndicators++;
            
            results.push_back({features, noiseIndicators >= 2});
        }
        
        return results;
    }
};

int main(int argc, char* argv[]) {
    std::string audioFilePath = argc >= 2 ? argv[1] : "C:/Users/Dell/Downloads/232198__nkroher__urban_night_rain_1.wav";

    try {
        FastAudioProcessor processor;
        auto audioData = processor.loadWavFile(audioFilePath);
        auto results = processor.analyzeAudio();

        // Calculate statistics
        int noiseCount = 0;
        for (const auto& result : results) {
            if (result.second) noiseCount++;
        }

        double noisePercentage = (results.size() > 0) ? 
            (noiseCount * 100.0 / results.size()) : 0.0;

        // Output results
        std::cout << "Audio Analysis Results\n"
                  << "=====================\n"
                  << "File: " << audioFilePath << "\n"
                  << "Total windows analyzed: " << results.size() << "\n";

        // Print first few results
        std::cout << "Sample windows analysis (first 5):\n"
                  << "--------------------------------\n"
                  << "RMS Energy\tZero Crossing Rate\tSpectral Centroid\t\n";
        
        for (size_t i = 0; i < std::min(size_t(10), results.size()); i++) {
            const auto& features = results[i].first;
            for (double feature : features) {
                std::cout << feature << "\t\t";
            }
            std :: cout<<"\n";
            //std::cout << (results[i].second ? "NOISE" : "CLEAN") << "\n";
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}