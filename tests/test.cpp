#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <iostream>

BOOST_AUTO_TEST_CASE(TestFail) {
  BOOST_CHECK_EQUAL(true, 0);
}
