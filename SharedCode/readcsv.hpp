#ifndef READCSV_HPP
#define READCSV_HPP

#include <vector>

#include "Point.hpp"
#include "csv.hpp"

// reads specified csv file at the three specified columns
std::vector<Point> readcsv(std::string cat1, std::string cat2, std::string cat3) {
    std::vector<Point> points;
    std::vector<Point> currentLine;
    csv::CSVReader reader("./tracks_features.csv");
    for (csv::CSVRow &row : reader) {
        points.push_back(Point(row[cat1].get<double>(), row[cat2].get<double>(),
                              row[cat3].get<double>()));
    }
    return points;
  }

#endif
