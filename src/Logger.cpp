// Copyright (c) 2018 Daniel dos Santos Marques <danielsmarques7@gmail.com>
// License: BSD 3 clause

#include "Logger.h"

#include <chrono>
#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

std::ostream Logger::out_(std::cout.rdbuf());
std::filebuf Logger::fb_;

void Logger::setOutput(std::string fileName) {
  if (fileName.compare("") == 0) {
    out_.set_rdbuf(std::cout.rdbuf());
  }
  else {
    fb_.open(fileName, std::ios::out);
    out_.set_rdbuf(&fb_);
  }
}


void Logger::closeOutput() {
  fb_.close();
}


std::ostream& Logger::log() {
  std::time_t now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  out_ << std::endl << std::put_time(std::localtime(&now), "%Y-%m-%d_%H-%M-%S") << ": ";
  return out_;
}
