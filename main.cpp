#include <iostream>
#include <vector>
#include <queue>
#include <unordered_map>
#include <set>
#include <deque>
#include <memory>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <string>

// WAV file header structure
struct WAVHeader {
    char riff[4];           // "RIFF"
    uint32_t fileSize;      // File size - 8
    char wave[4];           // "WAVE"
    char fmt[4];            // "fmt "
    uint32_t fmtSize;       // Format chunk size
    uint16_t audioFormat;   // Audio format (1 = PCM)
    uint16_t numChannels;   // Number of channels
    uint32_t sampleRate;    // Sample rate
    uint32_t byteRate;      // Byte rate
    uint16_t blockAlign;    // Block align
    uint16_t bitsPerSample; // Bits per sample
};

class AudioProcessor {
private:
    std::vector<double> audioData;
    const int windowSize = 2048;
    const int hopSize = 512;
    uint32_t sampleRate;

public:
    std::vector<double> loadWavFile(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open WAV file");
        }

        // Read WAV header
        WAVHeader header;
        file.read(reinterpret_cast<char*>(&header), sizeof(WAVHeader));

        // Verify WAV format
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
            
            if (std::string(chunkID, 4) == "data") {
                break;
            }
            file.seekg(chunkSize, std::ios::cur);
        }

        // Read audio data
        std::vector<int16_t> rawData(chunkSize / 2);
        file.read(reinterpret_cast<char*>(rawData.data()), chunkSize);

        // Convert to mono if stereo and normalize to [-1, 1]
        audioData.clear();
        sampleRate = header.sampleRate;
        
        for (size_t i = 0; i < rawData.size(); i += header.numChannels) {
            double sample = 0;
            for (int ch = 0; ch < header.numChannels; ch++) {
                sample += rawData[i + ch];
            }
            sample /= (header.numChannels * 32768.0); // Normalize
            audioData.push_back(sample);
        }

        return audioData;
    }

    std::vector<std::vector<double>> extractPatterns() {
        std::vector<std::vector<double>> patterns;
        
        // Process audio in windows
        for (size_t i = 0; i < audioData.size() - windowSize; i += hopSize) {
            std::vector<double> window(audioData.begin() + i, 
                                     audioData.begin() + i + windowSize);
            
            // Apply Hanning window
            for (size_t j = 0; j < window.size(); j++) {
                window[j] *= 0.5 * (1 - cos(2 * M_PI * j / (windowSize - 1)));
            }
            
            // Extract features
            std::vector<double> features = extractFeatures(window);
            
            if (!features.empty()) {
                patterns.push_back(features);
            }
        }
        
        return patterns;
    }

private:
    std::vector<double> extractFeatures(const std::vector<double>& window) {
        std::vector<double> features;
        
        // RMS Energy
        double rms = 0;
        for (double sample : window) {
            rms += sample * sample;
        }
        rms = sqrt(rms / window.size());
        features.push_back(rms);
        
        // Zero Crossing Rate
        int zcr = 0;
        for (size_t i = 1; i < window.size(); i++) {
            if ((window[i] >= 0 && window[i-1] < 0) || 
                (window[i] < 0 && window[i-1] >= 0)) {
                zcr++;
            }
        }
        features.push_back(static_cast<double>(zcr) / window.size());
        
        // Spectral Centroid (simplified)
        std::vector<double> spectrum = computeSimpleSpectrum(window);
        double weightedSum = 0.0;
        double sum = 0.0;
        for (size_t i = 0; i < spectrum.size(); i++) {
            weightedSum += i * spectrum[i];
            sum += spectrum[i];
        }
        double centroid = sum != 0 ? weightedSum / sum : 0;
        features.push_back(centroid);
        
        return features;
    }

    std::vector<double> computeSimpleSpectrum(const std::vector<double>& signal) {
        std::vector<double> spectrum(signal.size() / 2);
        for (size_t k = 0; k < spectrum.size(); k++) {
            double real = 0, imag = 0;
            for (size_t n = 0; n < signal.size(); n++) {
                double angle = 2 * M_PI * k * n / signal.size();
                real += signal[n] * cos(angle);
                imag -= signal[n] * sin(angle);
            }
            spectrum[k] = sqrt(real * real + imag * imag);
        }
        return spectrum;
    }
};

// Rest of the PatternAnalyzer class remains the same...
struct CompareFrequency {
    bool operator()(const std::pair<double, int>& a, const std::pair<double, int>& b) {
        return a.first < b.first;
    }
};

// Trie node for pattern matching
class TrieNode {
public:
    std::unordered_map<char, std::unique_ptr<TrieNode>> children;
    bool isEndOfPattern;
    std::vector<double> pattern;

    TrieNode() : isEndOfPattern(false) {}
};

// AVL Tree node for balanced pattern storage
class AVLNode {
public:
    double frequency;
    int height;
    std::unique_ptr<AVLNode> left, right;
    std::vector<double> pattern;

    AVLNode(double freq) : frequency(freq), height(1) {}
};


class UnionFind {
private:
    std::unordered_map<int, int> parent;
    std::unordered_map<int, int> rank;

public:
    int find(int x) {
        if (parent.find(x) == parent.end()) {
            parent[x] = x;
            rank[x] = 0;
            return x;
        }
        
        if (parent[x] != x) {
            parent[x] = find(parent[x]); 
        }
        return parent[x];
    }

    void unite(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return;

        if (rank[px] < rank[py]) {
            parent[px] = py;
        } else if (rank[px] > rank[py]) {
            parent[py] = px;
        } else {
            parent[py] = px;
            rank[px]++;
        }
    }
};

class PatternAnalyzer {
private:
    std::unique_ptr<TrieNode> patternTrie;
    std::unique_ptr<AVLNode> frequencyTree;
    std::priority_queue<std::pair<double, int>, 
                       std::vector<std::pair<double, int>>, 
                       CompareFrequency> frequencyHeap;
    std::deque<std::vector<double>> recentPatterns;
    UnionFind patternClusters;
    
    std::unordered_map<std::string, double> similarityCache;

    // Added: Insert pattern into trie
    void insertPatternIntoTrie(const std::vector<double>& pattern) {
        TrieNode* current = patternTrie.get();
        
        // Convert pattern to discrete steps for trie insertion
        for (double value : pattern) {
            // Convert double to char representation for trie
            char discreteValue = static_cast<char>(std::min(255.0, std::max(0.0, value / 10)));
            
            if (!current->children[discreteValue]) {
                current->children[discreteValue] = std::make_unique<TrieNode>();
            }
            current = current->children[discreteValue].get();
        }
        
        current->isEndOfPattern = true;
        current->pattern = pattern;
    }

    int getHeight(AVLNode* node) {
        return node ? node->height : 0;
    }

    int getBalance(AVLNode* node) {
        return node ? getHeight(node->left.get()) - getHeight(node->right.get()) : 0;
    }

    void updateHeight(AVLNode* node) {
        if (node) {
            node->height = 1 + std::max(getHeight(node->left.get()), 
                                      getHeight(node->right.get()));
        }
    }

    std::unique_ptr<AVLNode> rightRotate(std::unique_ptr<AVLNode> y) {
        if (!y || !y->left) return y;
        
        auto x = std::move(y->left);
        auto T2 = std::move(x->right);

        x->right = std::move(y);
        x->right->left = std::move(T2);

        updateHeight(x->right.get());
        updateHeight(x.get());

        return x;
    }

    std::unique_ptr<AVLNode> leftRotate(std::unique_ptr<AVLNode> x) {
        if (!x || !x->right) return x;
        
        auto y = std::move(x->right);
        auto T2 = std::move(y->left);

        y->left = std::move(x);
        y->left->right = std::move(T2);

        updateHeight(y->left.get());
        updateHeight(y.get());

        return y;
    }

    double calculatePatternSimilarity(const std::vector<double>& pattern1, 
                                    const std::vector<double>& pattern2) {
        int m = pattern1.size(), n = pattern2.size();
        std::vector<std::vector<double>> dp(m + 1, std::vector<double>(n + 1, 0));

        for (int i = 0; i <= m; i++) dp[i][0] = i;
        for (int j = 0; j <= n; j++) dp[0][j] = j;

        for (int i = 1; i <= m; i++) {
            for (int j = 1; j <= n; j++) {
                double diff = std::abs(pattern1[i-1] - pattern2[j-1]);
                if (diff < 0.1) {
                    dp[i][j] = dp[i-1][j-1];
                } else {
                    dp[i][j] = 1 + std::min({dp[i-1][j], dp[i][j-1], dp[i-1][j-1]});
                }
            }
        }

        return 1.0 - (dp[m][n] / std::max(m, n));
    }

public:
    PatternAnalyzer() : patternTrie(std::make_unique<TrieNode>()) {}

    void insertPattern(const std::vector<double>& pattern, double frequency) {
        frequencyTree = insertAVL(std::move(frequencyTree), frequency, pattern);
        frequencyHeap.push({frequency, frequencyHeap.size()});
        
        if (recentPatterns.size() >= 1000) {
            recentPatterns.pop_front();
        }
        recentPatterns.push_back(pattern);
        insertPatternIntoTrie(pattern);
    }

    std::unique_ptr<AVLNode> insertAVL(std::unique_ptr<AVLNode> node, 
                                      double frequency,
                                      const std::vector<double>& pattern) {
        if (!node) {
            auto newNode = std::make_unique<AVLNode>(frequency);
            newNode->pattern = pattern;
            return newNode;
        }

        if (frequency < node->frequency) {
            node->left = insertAVL(std::move(node->left), frequency, pattern);
        } else {
            node->right = insertAVL(std::move(node->right), frequency, pattern);
        }

        updateHeight(node.get());
        int balance = getBalance(node.get());

        // Left Left Case
        if (balance > 1 && frequency < node->left->frequency) {
            return rightRotate(std::move(node));
        }

        // Right Right Case
        if (balance < -1 && frequency > node->right->frequency) {
            return leftRotate(std::move(node));
        }

        // Left Right Case
        if (balance > 1 && frequency > node->left->frequency) {
            node->left = leftRotate(std::move(node->left));
            return rightRotate(std::move(node));
        }

        // Right Left Case
        if (balance < -1 && frequency < node->right->frequency) {
            node->right = rightRotate(std::move(node->right));
            return leftRotate(std::move(node));
        }

        return node;
    }

    std::vector<std::vector<double>> findSimilarPatterns(const std::vector<double>& pattern, 
                                                        double similarityThreshold = 0.8) {
        std::vector<std::vector<double>> result;
        
        for (const auto& storedPattern : recentPatterns) {
            double similarity = calculatePatternSimilarity(pattern, storedPattern);
            if (similarity >= similarityThreshold) {
                result.push_back(storedPattern);
            }
        }

        return result;
    }

    std::vector<std::vector<std::vector<double>>> clusterPatterns(double similarityThreshold = 0.8) {
        std::unordered_map<int, std::vector<std::vector<double>>> clusters;
        
        for (size_t i = 0; i < recentPatterns.size(); i++) {
            for (size_t j = i + 1; j < recentPatterns.size(); j++) {
                double similarity = calculatePatternSimilarity(recentPatterns[i], recentPatterns[j]);
                if (similarity >= similarityThreshold) {
                    patternClusters.unite(i, j);
                }
            }
        }

        for (size_t i = 0; i < recentPatterns.size(); i++) {
            int root = patternClusters.find(i);
            clusters[root].push_back(recentPatterns[i]);
        }

        std::vector<std::vector<std::vector<double>>> result;
        for (const auto& cluster : clusters) {
            result.push_back(cluster.second);
        }

        return result;
    }

    // Added: Print pattern information for debugging
    void printPatternInfo() const {
        std::cout << "Number of recent patterns: " << recentPatterns.size() << "\n";
        if (!recentPatterns.empty()) {
            std::cout << "Latest pattern size: " << recentPatterns.back().size() << " values\n";
        }
    }
};

int main(int argc, char* argv[]) {
    std::string audioFilePath = "C:/Users/Dell/Desktop/101415-3-0-2 (1).wav"; // Default file path

    if (argc >= 2) {
        audioFilePath = argv[1]; // If a file path is provided as an argument, use it
    }

    try {
        // Process audio file
        AudioProcessor processor;
        auto audioData = processor.loadWavFile(audioFilePath); // Use the selected file path
        auto patterns = processor.extractPatterns();

        // Analyze patterns
        PatternAnalyzer analyzer;
        for (const auto& pattern : patterns) {
            analyzer.insertPattern(pattern, pattern[0]); // Use first feature as frequency
        }

        // Cluster patterns
        auto clusters = analyzer.clusterPatterns(0.8);

        // Print results
        std::cout << "Analysis complete.\n";
        std::cout << "Found " << patterns.size() << " patterns\n";
        std::cout << "Grouped into " << clusters.size() << " clusters\n";

        // Print pattern features
        std::cout << "\nSample pattern features:\n";
        if (!patterns.empty()) {
            std::cout << "RMS Energy\tZero Crossing Rate\tSpectral Centroid\n";
            for (size_t i = 0; i < std::min(size_t(5), patterns.size()); i++) {
                for (double feature : patterns[i]) {
                    std::cout << feature << "\t";
                }
                std::cout << "\n";
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
