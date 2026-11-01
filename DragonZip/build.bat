@echo off
rem DragonZip — Windows uchun yig'ish (MinGW-w64 g++ kerak: https://winlibs.com)
g++ -O2 -std=c++17 -pthread dragonzip.cpp -o dragonzip.exe -static -lshell32
if %errorlevel% neq 0 (
  echo Xato: yig'ib bo'lmadi. g++ o'rnatilganini tekshiring.
  exit /b 1
)
echo Tayyor: dragonzip.exe
