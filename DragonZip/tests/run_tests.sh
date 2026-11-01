#!/usr/bin/env bash
# DragonZip uchun avtomatik sinovlar: ./tests/run_tests.sh [dragonzip-binari]
set -u
DZ="${1:-./dragonzip}"
DZ="$(cd "$(dirname "$DZ")" && pwd)/$(basename "$DZ")"
W="$(mktemp -d)"
trap 'rm -rf "$W"' EXIT
pass=0; fail=0
ok()   { echo "  [OK]   $1"; pass=$((pass+1)); }
bad()  { echo "  [XATO] $1"; fail=$((fail+1)); }
check(){ if eval "$2" >/dev/null 2>&1; then ok "$1"; else bad "$1"; fi; }

echo "DragonZip sinovlari ($DZ)"
cd "$W"
mkdir -p src/papka/ichki "src/bo'sh papka" "src/Ўзбек тили"
python3 - <<'PY'
import os, random
random.seed(42)
words = "salom dunyo kitob maktab o'quvchi ustoz dastur kompyuter siqish arxiv algoritm ma'lumot".split()
txt = " ".join(random.choice(words) for _ in range(120000))
open("src/matn.txt", "w").write(txt)
open("src/papka/kod.py", "w").write("".join(f"def f{i}(x):\n    return x * {i} + {i % 7}\n\n" for i in range(8000)))
open("src/papka/ichki/random.bin", "wb").write(os.urandom(300000))
open("src/papka/ichki/nollar.bin", "wb").write(bytes(500000))
open("src/bo'sh.txt", "wb").write(b"")
open("src/bir.txt", "wb").write(b"A")
open("src/Ўзбек тили/салом дунё.txt", "w", encoding="utf-8").write("Assalomu alaykum! Ўзбекистон 🌙\n" * 2000)
open("src/fayl bo'shliq bilan.csv", "w").write("".join(f"{i},{i*i},{i%13},nom_{i}\n" for i in range(20000)))
PY
cp /bin/true "src/papka/dastur.elf" 2>/dev/null || cp "$DZ" "src/papka/dastur.elf"
touch -d "2020-01-02 03:04:05" src/matn.txt

# 1) Ko'p blokli, ko'p oqimli siqish va ochish
check "arxiv yaratish (-t 4 -b 1)" "'$DZ' a -q -t 4 -b 1 arx.drz src"
check "arxivni tekshirish (t)" "'$DZ' t -q arx.drz"
check "ro'yxat (l)" "'$DZ' l arx.drz | grep -q 'salom dunyo\|салом дунё'"
check "ochish (x)" "'$DZ' x -q -o out arx.drz"
check "barcha fayllar bitma-bit bir xil" "diff -r src out/src"
check "bo'sh papka tiklandi" "test -d \"out/src/bo'sh papka\""
check "bo'sh fayl tiklandi" "test -f \"out/src/bo'sh.txt\" && test ! -s \"out/src/bo'sh.txt\""
check "o'zgartirilgan vaqt saqlandi" "test \"\$(stat -c %Y src/matn.txt)\" = \"\$(stat -c %Y out/src/matn.txt)\""
check "ruxsatlar saqlandi" "test \"\$(stat -c %a src/papka/dastur.elf)\" = \"\$(stat -c %a out/src/papka/dastur.elf)\""
check "tasodifiy fayl 'saqlangan' deb belgilandi" "'$DZ' l arx.drz | grep random.bin | grep -q saqlangan"
check "dastur fayliga x86 filtri qo'llandi" "'$DZ' l arx.drz | grep dastur.elf | grep -q x86"
orig=$(du -sb src | cut -f1); arc=$(stat -c %s arx.drz)
echo "         asl: $orig bayt -> arxiv: $arc bayt"

# 2) Bitta blok, bitta oqim, boshqa daraja
check "bitta oqim, -3 daraja" "'$DZ' a -q -3 -t 1 arx1.drz src/matn.txt src/papka && '$DZ' x -q -o out1 arx1.drz && cmp src/matn.txt out1/matn.txt && diff -r src/papka out1/papka"
check "--no-detect bilan" "'$DZ' a -q --no-detect arx2.drz src/papka/ichki && '$DZ' x -q -o out2 arx2.drz && diff -r src/papka/ichki out2/ichki"

# 3) Himoya: ustidan yozish
check "mavjud arxiv -f siz rad etiladi" "! '$DZ' a -q arx.drz src/bir.txt"
check "mavjud fayllar -f siz rad etiladi" "! '$DZ' x -q -o out arx.drz"
check "-f bilan qayta ochish" "'$DZ' x -q -f -o out arx.drz && diff -r src out/src"

# 4) Buzilgan arxivlarni aniqlash
cp arx.drz buz.drz
python3 -c "
import sys; d=bytearray(open('buz.drz','rb').read()); d[len(d)//3]^=0x55; open('buz.drz','wb').write(d)"
check "buzilgan bayt aniqlanadi" "! '$DZ' t -q buz.drz"
head -c $(( $(stat -c %s arx.drz) - 100 )) arx.drz > kesik.drz
check "kesilgan arxiv aniqlanadi" "! '$DZ' t -q kesik.drz"
echo "bu arxiv emas" > soxta.drz
check "arxiv bo'lmagan fayl rad etiladi" "! '$DZ' l soxta.drz"

# 5) Xavfli yo'l ("../") bilan yasalgan arxiv rad etiladi
python3 - <<'PY'
import struct, zlib
def var(v):
    o=b""
    while v>=128: o+=bytes([(v&127)|128]); v>>=7
    return o+bytes([v])
data=b"zarar"
d=b"D1"+var(1)+bytes([0,0])+var(8)+b"../evilx"+var(len(data))+var(0)+var(0o644)+struct.pack("<I",zlib.crc32(data))
def blk(tag,raw):
    h=bytes([ord(tag),0,0,0])+struct.pack("<IQQ",zlib.crc32(raw),len(raw),len(raw))
    return h+struct.pack("<I",zlib.crc32(h))+raw
out=b"DRGZ"+bytes([1,6,0,0])+bytes(8)
out+=blk('B',data)
off=len(out)
out+=blk('D',d)
t=struct.pack("<QI",off,1)
out+=t+struct.pack("<I",zlib.crc32(t))+b"DRGZ-END"
open("yomon.drz","wb").write(out)
PY
mkdir -p xavf/ichki
check "'../' yo'li rad etiladi" "(cd xavf/ichki && '$DZ' x -q ../../yomon.drz 2>&1 | grep -q \"xavfli yo'l\")"
check "arxivdan tashqariga hech narsa yozilmadi" "test ! -e xavf/evilx"

echo
echo "Natija: $pass ta muvaffaqiyatli, $fail ta xato"
[ "$fail" -eq 0 ]
