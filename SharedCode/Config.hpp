
#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <string>
#include <iostream>

struct Config {
  std::string category1;
  std::string category2;
  std::string category3;
  bool writeToFile;
  int k;
  int epochs;
  int numThreads;

  Config() : category1("danceablility"), category2("loudness"), category3("instrumentalness"), k(1), epochs(10), writeToFile(false), numThreads(1) {}

  bool parseInput(int argc, char* argv[]) {
    // no arguments provided, output the default configuration
    if (argc == 1) {
        std::cout << "No input parameters provided, using defaults:" << std::endl;
        std::cout << "   category 1  : " << category1 << std::endl;
        std::cout << "   category 2  : " << category2 << std::endl;
        std::cout << "   category 3  : " << category3 << std::endl;
        std::cout << "   writeToFile : " << (writeToFile ? "true" : "false") << std::endl;
        std::cout << "   k           : " << k << std::endl;
        std::cout << "   epochs      : " << epochs << std::endl;
        std::cout << "   numThreads  : " << numThreads << std::endl;
        return true;
    }
    if (argc == 7 || argc == 8) // case of valid argument counts
    {
        category1 = argv[1];
        category2 = argv[2];
        category3 = argv[3];
        writeToFile = (argv[4][0] == 't' || argv[4][0] == 'T');
        k = std::stoi(argv[5]);
        epochs = std::stoi(argv[6]);

        std::cout << "Using the following parameters provided by user:" << std::endl;
        std::cout << "   category 1  : " << category1 << std::endl;
        std::cout << "   category 2  : " << category2 << std::endl;
        std::cout << "   category 3  : " << category3 << std::endl;
        std::cout << "   writeToFile : " << (writeToFile ? "true" : "false") << std::endl;
        std::cout << "   k           : " << k << std::endl;
        std::cout << "   epochs      : " << epochs << std::endl;
        if (argc == 8)
        {
            numThreads = std::stoi(argv[7]);
            std::cout << "   numThreads  : " << numThreads << std::endl;
        }
        return true;
    }
    // print out usage in case of invalid input parameters
    std::cout << "Invalid input parameters" << std::endl;
    std::cout << "Usage: ./ExecutableName <category 1> <category 2> <category 3> <writeToFile (t/f)> <k> <epochs> <numThreads (optional)>" << std::endl;
    std::cout << "Usage: ./ExecutableName            runs with default configuration" << std::endl;
    return false;
  }
};

#endif
