#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
	cout << "End of the world!" << endl;

	if(argc > 1) {
		cout << "This program will never take any arguments!" << endl;
	}
}
