#include <iostream>
#include <string>

using namespace std;

int main(int argc, char* argv[]) {
	for(int i = 0; i < 10; ++i) {
		cout << i * 3 << endl;
        }
                
	cout << "End of the world!" << endl;

	if(argc > 1) {
		cout << "This program will never take any arguments!" << endl;
	}

	cout << "Hello group!" << endl;
}
