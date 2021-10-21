int main() {
    volatile char* ptr = new char[100];
    delete[] ptr;
}
