#include "Sample.h"

Sample::Sample(int totAttributes, int totClasses) {
  inxValue_.resize(totAttributes);
  benefit_.resize(totClasses);
}
