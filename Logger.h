#pragma once
#include <string>

class Logger {
public:
  static void setOutput(std::string fileName);
  static void closeOutput();
  static std::ostream& log();

private:
  static std::ostream out_;
  static std::filebuf fb_;
};
