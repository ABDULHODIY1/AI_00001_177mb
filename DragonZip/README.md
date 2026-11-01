# 🐉 DragonZip — kuchli, yo'qotishsiz arxivator

DragonZip — ZIP ga o'xshash arxivator (fayl va papkalarni bitta arxivga
joylaydi). Lekin u ZIP dagi oddiy algoritm (Deflate) o'rniga **context
mixing (CM)** algoritmidan foydalanadi. Bu dunyodagi eng kuchli siqish
dasturlari (PAQ, cmix) ishlatadigan usul.

* **Yo'qotishsiz (lossless):** ochilgan fayl asl fayl bilan bitma-bit bir xil
  bo'ladi. Har bir fayl va blok CRC32 bilan tekshiriladi.
* **Kuchli:** sinovlarda ZIP dan o'rtacha **45% kichik**, `xz -9e` (7-Zip/LZMA)
  dan **29% kichik**, `bzip2 -9` dan **33% kichik** arxiv hosil qildi (natijalar
  pastda).
* **Bitta fayl, tashqi kutubxonalarsiz:** faqat C++17 kompilyator kerak.
  Linux, macOS, Windows va Android (Termux) da ishlaydi.

---

## Natijalar (haqiqiy o'lchovlar)

Har bir fayl alohida siqilgan. Qiymatlar baytlarda, kichigi yaxshiroq.

| Fayl | Turi | Asl hajm | ZIP -9 | bzip2 -9 | xz -9e | **DragonZip** | ZIP ga nisbatan |
|---|---|---:|---:|---:|---:|---:|---:|
| binary_bash.elf | x86-64 dastur (bash) | 1 446 024 | 691 743 | 649 682 | 583 796 | **429 312** | 38% kichik |
| binary_git.elf | x86-64 dastur (git) | 4 066 232 | 2 036 931 | 1 826 472 | 1 682 932 | **1 190 261** | 42% kichik |
| book.txt | Ingliz kitobi (Newton, Opticks) | 567 198 | 173 653 | 132 469 | 143 496 | **105 007** | 40% kichik |
| data.json | JSON ma'lumot | 498 094 | 56 918 | 43 889 | 43 844 | **30 143** | 47% kichik |
| data.xml | XML ma'lumot | 1 016 601 | 109 808 | 91 679 | 83 040 | **43 262** | 61% kichik |
| source_py.txt | Python kodi | 6 772 427 | 1 470 382 | 1 155 835 | 1 058 952 | **762 558** | 48% kichik |
| table.csv | CSV jadval (xeshlar) | 342 934 | 207 186 | 183 301 | 191 552 | **167 202** | 19% kichik |
| text_vimdoc.txt | Texnik hujjatlar (matn) | 9 932 585 | 2 872 677 | 2 173 746 | 2 114 284 | **1 484 058** | 48% kichik |
| **Jami** | | 24 642 095 | 7 619 298 | 6 257 073 | 5 901 896 | **4 211 803** | 45% kichik |

Siqish koeffitsienti: ZIP — 3.23×, bzip2 — 3.94×, xz — 4.18×, **DragonZip — 5.85×**.

Butun papka bitta arxivga (8 fayl, 23.5 MB, 4 yadroli Xeon 2.8 GHz):

| Dastur | Arxiv hajmi | Vaqt |
|---|---:|---:|
| `zip -9 -r` | 7 619 270 | 2.7 s |
| `tar \| bzip2 -9` | 6 295 763 | 2.3 s |
| `tar \| xz -9e` | 5 862 804 | 21 s |
| `dragonzip a -t 4 -b 6` (4 oqim) | 4 250 411 | 42 s |
| **`dragonzip a`** (solid, 1 blok) | **4 202 438** | 1 m 39 s |

Ochish/tekshirish (4 oqim): 30 s.

### Halol ogohlantirishlar

* **Hech bir yo'qotishsiz algoritm hamma narsani siqa olmaydi.** Bu matematik
  qonun ("kaptarlar uyasi" prinsipi). Allaqachon siqilgan fayllar (JPG, PNG,
  MP4, MP3, ZIP, RAR, 7z…) deyarli kichraymaydi. DragonZip bunday fayllarni
  avtomatik aniqlaydi va vaqtni behuda sarflamaslik uchun ularni o'zgartirmasdan
  saqlaydi.
* **Kuch tezlik hisobiga keladi.** DragonZip har bir *bit* uchun o'nlab
  modellarni hisoblaydi, shuning uchun u bitta oqimda taxminan **0.2–0.35 MB/s**
  tezlikda ishlaydi (ZIP dan ancha sekin). Ochish tezligi ham taxminan shunday. Katta hajmlar uchun bir nechta yadro
  ishlatiladi (`-t`).
* Siqish darajasi ma'lumot turiga bog'liq: matn, kod, jadval, JSON/XML va
  dasturlar juda yaxshi siqiladi; tasodifiy yoki shifrlangan ma'lumot esa
  siqilmaydi.

---

## Yig'ish (kompilyatsiya)

### Linux / macOS
```bash
cd DragonZip
make            # yoki: g++ -O2 -std=c++17 -pthread dragonzip.cpp -o dragonzip
make native     # shu protsessorga moslab (AVX2 bo'lsa biroz tezroq)
make test       # avtomatik sinovlar
```

### Windows
[MinGW-w64](https://winlibs.com) o'rnating va `build.bat` ni ishga tushiring, yoki:
```bat
g++ -O2 -std=c++17 -pthread dragonzip.cpp -o dragonzip.exe -static -lshell32
```
Visual Studio bilan: `cmake -S . -B build && cmake --build build --config Release`

### Android (Termux)
```bash
pkg install clang make
make
```

> Barcha hisoblar butun sonlarda bajariladi, shuning uchun turli kompilyator va
> protsessorlarda yig'ilgan dasturlar bir-birining arxivini ocha oladi. SIMD (AVX2/SSE2)
> va oddiy C++ variantlari bitma-bit bir xil natija beradi (x86-64 da GCC va Clang
> bilan tekshirilgan).

---

## Foydalanish

```bash
# Arxiv yaratish (fayllar va papkalar)
./dragonzip a arxiv.drz hujjatlar/ rasm.bmp eslatma.txt

# Maksimal siqish (1 GB xotira, bitta solid blok)
./dragonzip a -9 -t 1 arxiv.drz papka/

# Tarkibini ko'rish
./dragonzip l arxiv.drz

# Butunligini tekshirish (hech narsa yozmasdan)
./dragonzip t arxiv.drz

# Ochish (joriy papkaga yoki -o bilan ko'rsatilgan papkaga)
./dragonzip x arxiv.drz -o natija/

# Siqish sinovi (fayl xotirada siqiladi, ochiladi va tekshiriladi)
./dragonzip bench kitob.txt
```

| Parametr | Ma'nosi |
|---|---|
| `-1` … `-9` | Model xotirasi: `-1` = 4 MB … `-6` = 128 MB (standart) … `-9` = 1 GB (har bir oqim uchun). Ko'proq xotira katta fayllarda yaxshiroq siqadi. |
| `-t N` | Oqimlar (yadrolar) soni. Standart: yadrolar soni, ko'pi bilan 4. |
| `-b N` | Blok hajmi (MB). Har bir blok alohida siqiladi; katta blok yaxshiroq siqadi, kichik blok parallellikni oshiradi. Standart: ma'lumot oqimlarga bo'linadi, lekin blok 64 MB dan kichik bo'lmaydi. |
| `-o PAPKA` | Ochish uchun papka. |
| `-f` | Mavjud fayllar ustidan yozishga ruxsat. |
| `-q` | Jim rejim. |
| `--no-detect` | Siqilgan fayllarni aniqlashni o'chirish (hammasi CM bilan siqiladi). |

Xotira sarfi taxminan: `oqimlar × (daraja xotirasi + 64 MB + 1.5 × blok)`.
Ochishda ham shuncha xotira kerak bo'ladi.

---

## Algoritm qanday ishlaydi?

ZIP takrorlangan qismlarni qidiradi va ularni havola bilan almashtiradi.
DragonZip esa butunlay boshqacha ishlaydi: u faylni **bitma-bit bashorat
qiladi**. Har bir bit uchun "bu bit 1 bo'lish ehtimoli qancha?" degan savolga
javob beradi va **arifmetik kodlovchi** shu ehtimollik asosida bitni yozadi.
Bashorat qanchalik aniq bo'lsa, bit shunchalik kam joy egallaydi (masalan,
99% aniq bashorat qilingan bit atigi 0.015 bit joy oladi).

```
 kirish bayti ──► 20 ta kontekst modeli ──► 5 ta neyron mikser ──► yakuniy mikser
                  + match model                (turli kontekstlar       │
                  + order-1/2 hisoblagichlar    bo'yicha og'irliklar)   ▼
                                                                    4 ta APM/SSE
                                                                        │
                                              arifmetik kodlovchi ◄─────┘
```

1. **Kontekst modellari (20 ta).** Har biri oldingi baytlarning turli
   "ko'rinishiga" qaraydi:
   * order-0 … order-8 (oxirgi 0–8 bayt),
   * so'z modellari (joriy so'z, oldingi 1–2 so'z bilan; katta-kichik harf farqsiz, UTF-8 harflari ham),
   * siyrak (sparse) kontekstlar — binar fayllar va jadvallar uchun,
   * matn ustuni (yuqoridagi qatordagi belgi) — kod va jadvallar uchun,
   * yozuv (record) modeli — takrorlanuvchi tuzilmalarni avtomatik aniqlaydi,
   * bilvosita (indirect) kontekstlar — "bu kontekstdan keyin odatda nima keladi".

   Har bir kontekst uchun **bit tarixi holati** (1 bayt) saqlanadi: necha marta
   0 va 1 kelgani, eski ma'lumotni asta-sekin "unutish" bilan. Holatlar 16 baytlik
   bloklarda (bitta yarim bayt = 15 bit-tugun) xesh jadvalda saqlanadi —
   bitta kesh qatorida 3 yo'lli qidiruv.
2. **Match model.** Fayl ichida oldin uchragan eng uzun mos ketma-ketlikni topadi
   va keyingi baytni bashorat qiladi (uzun takrorlar uchun juda kuchli).
3. **Neyron mikser.** Barcha modellarning bashoratlari logistik (stretch) domenda
   og'irliklar bilan qo'shiladi; og'irliklar har bitdan keyin onlayn gradient
   tushish bilan o'rganiladi. 5 ta mikser turli kontekstlar bo'yicha o'z
   og'irliklar to'plamini tanlaydi, keyin yakuniy mikser ularni birlashtiradi.
4. **APM/SSE.** Yakuniy ehtimollik 4 ta adaptiv xarita orqali aniqlashtiriladi.
5. **Arifmetik kodlovchi** — 32 bitli, 16 bitli ehtimollik aniqligi bilan.

Qo'shimcha:
* **x86 E8/E9 filtri:** `.exe`, `.dll`, ELF dasturlaridagi CALL/JMP manzillarini
  nisbiydan mutlaq manzilga aylantiradi — sinovda dastur fayli ~10% yaxshiroq siqildi.
* **Solid arxiv:** fayllar kengaytmasi bo'yicha tartiblanib, bitta oqimga
  birlashtiriladi — o'xshash fayllar bir-biridan "o'rganadi".
* **Siqilgan fayllarni aniqlash:** entropiya tahlili (JPG/MP4/ZIP… → saqlanadi).

## Xavfsizlik va ishonchlilik

* Har bir blok, blok sarlavhasi, arxiv oxiri va har bir fayl CRC32 bilan tekshiriladi.
* Buzilgan yoki kesilgan arxivlar aniqlanadi (sinovda 150 ta tasodifiy buzilishning hammasi aniqlandi).
* `../`, mutlaq yo'llar va boshqa xavfli yo'llar ochishda rad etiladi (zip-slip himoyasi).
* Arxiv avval `.tmp` faylga yoziladi va faqat muvaffaqiyatli tugaganda nomi o'zgartiriladi.
* Fayl vaqtlari, ruxsatlari va bo'sh papkalar tiklanadi.

## Arxiv formati (.drz)

```
[Sarlavha 16 B]   "DRGZ" | versiya | daraja | zaxira
[Ma'lumot bloklari]
    'B' | usul (0=saqlash, 1=CM) | jadval bitlari | 0 | CRC32 | asl hajm (8) | siqilgan hajm (8) | sarlavha CRC32 | ma'lumot
[Katalog bloki]   xuddi shu format, 'D' belgisi bilan (fayl nomlari, hajmlar, vaqtlar, CRC lar)
[Yakun 24 B]      katalog joyi (8) | bloklar soni (4) | CRC32 (4) | "DRGZ-END"
```

## Sinovlar

```bash
make test
```
Sinovlar: ko'p blokli/ko'p oqimli siqish, bo'sh fayl va papkalar, kirill/UTF-8
nomlar, siqilgan fayllarni aniqlash, x86 filtri, vaqt va ruxsatlarni tiklash,
ustidan yozish himoyasi, buzilgan/kesilgan arxivlarni aniqlash va `../` hujumi.
