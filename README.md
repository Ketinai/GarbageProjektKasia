https://arxiv.org/pdf/2104.00298.pdf
EfficientNetV2: Smaller Models and Faster Training - Mingxing Tan 1
Quoc V. Le 1
# https://medium.com/analytics-vidhya/convolution-operations-in-cnn-deep-learning-compter-vision-128906ece7d3
Medium Understanding “convolution” operations in CNN

Biblioteki języka Python ułatwiają nam obsługę danych i wykonywanie typowych oraz złożonych zadań za pomocą pojedynczej linii kodu.
Pandas – Ta biblioteka pomaga załadować ramkę danych w formacie tablicy 2D i posiada wiele funkcji do wykonywania zadań analizy za jednym zamachem.
Numpy – Tablice Numpy są bardzo szybkie i mogą wykonywać duże obliczenia w bardzo krótkim czasie.
Matplotlib – Ta biblioteka służy do tworzenia wizualizacji.
Sklearn – Ten moduł zawiera wiele bibliotek z funkcjami predefiniowanymi do wykonywania zadań od przetwarzania danych po rozwój i ocenę modeli.
OpenCV – Jest to biblioteka o otwartym kodzie głównie skupiona na przetwarzaniu i obsłudze obrazów.
TensorFlow – Jest to biblioteka open-source, która jest używana do uczenia maszynowego i sztucznej inteligencji oraz zapewnia szereg funkcji do osiągania złożonych funkcjonalności za pomocą pojedynczych linii kodu.
Od tego kroku będziemy korzystać z biblioteki TensorFlow, aby zbudować nasz model CNN. Framework Keras biblioteki TensorFlow zawiera wszystkie funkcjonalności, których można potrzebować do zdefiniowania architektury sieci neuronowej konwolucyjnej i jej trenowania na danych.

Batch_size
W praktyce zaobserwowano, że korzystając z większej partii danych, jakość modelu ulega pogorszeniu, co mierzy się jego zdolnością do generalizacji [...]
Metody wykorzystujące duże partie danych mają tendencję do zbiegania się do ostrych minimów funkcji treningowych i testowych—i jak dobrze wiadomo, ostre minima prowadzą do gorszej generalizacji. W przeciwieństwie do tego, metody wykorzystujące małe partie danych konsekwentnie zbiegają się do płaskich minimów, a nasze eksperymenty wspierają powszechnie przyjmowany pogląd, że wynika to z inherentnego szumu w estymacji gradientu.
Używamy mini batch-size’u:
Jeśli masz duży zestaw treningowy, użyj mini-batch gradient descent. W przeciwnym razie, dla małego zestawu treningowego, użyj batch gradient descent.
Rozmiary mini-partii często są wybierane jako potęga liczby 2, tj. 16, 32, 64, 128, 256 itd.
Podczas wybierania odpowiedniego rozmiaru dla mini-batch gradient descent, upewnij się, że mini-partia mieści się w pamięci CPU/GPU.
32 jest zazwyczaj domyślnym (dobrym) wyborem.
Mini batch_size wpływa na:
Czas treningu do zbieżności (ang. convergence, konwergencji): Wydaje się, że istnieje pewne optymalne rozwiązanie. Jeśli rozmiar partii (ang. batch_size) jest bardzo mały (np. 8), ten czas wzrasta. Jeśli batch_size  jest ogromny, jest on również wyższy niż minimum.
Czas treningu na epokę: Większe partie obliczeń są szybsze (są bardziej wydajne).
Jakość uzyskanego modelu: Im niższa, tym lepsza ze względu na lepszą generalizację (?).
Warto zauważyć interakcje hiperparametrów: Rozmiar partii może oddziaływać na inne hiperparametry, zwłaszcza na współczynnik uczenia. W niektórych eksperymentach to oddziaływanie może utrudniać wyizolowanie efektu samego rozmiaru partii na jakość modelu. Inne silne oddziaływanie występuje z wczesnym zatrzymywaniem dla regularyzacji.

target_size = (224, 224) oznacza, że rozmiar docelowy obrazów zostanie ustawiony na 224 piksele szerokości i 224 piksele wysokości. Jest to często używane, gdy przetwarzasz obrazy w sieciach neuronowych, na przykład podczas przeskalowywania obrazów do określonego rozmiaru przed podaniem ich do modelu. W tym przypadku obrazy zostaną przeskalowane lub przycięte, aby pasowały do określonego rozmiaru (224x224 pikseli), co ułatwia ich przetwarzanie w modelu.

EfficientNetV2B1
# https://medium.com/analytics-vidhya/convolution-operations-in-cnn-deep-learning-compter-vision-128906ece7d3
Medium Understanding “convolution” operations in CNN
https://arxiv.org/pdf/2104.00298.pdf
EfficientNetV2: Smaller Models and Faster Training - Mingxing Tan 1
Quoc V. Le 1
input_shape=(224,224,3)
Channels Last. Dane obrazowe są reprezentowane w trójwymiarowej tablicy przestrzeni (ang. three-dimensional array), gdzie ostatni kanał reprezentuje kanały kolorów, np. [wiersze][kolumny][kanały].
ustawimy wagi = 'imagenet', model uczenia maszynowego zaczyna działać, wykorzystując wagi wytrenowane przez model wytrenowany na zestawie danych ImageNet. W tym scenariuszu możliwe jest uzyskanie bardziej udanych wyników bez marnowania czasu na procesy propagacji wstecznej (Yildiz, 2023). Oznacza to, że nie tylko metryki sukcesu przewidywań wzrastają, ale także uzyskiwane są dokładniejsze wyniki przewidywań dla nowych obrazów przy użyciu opracowanego modelu (zauważyłem to podczas wielu doświadczeń).
.trainable = False oznacza, że wagi danego modelu lub warstwy nie będą trenowane podczas procesu uczenia. Innymi słowy, wagi te pozostaną stałe i nie ulegną aktualizacji podczas trenowania modelu na nowych danych. Jest to przydatne w przypadku wykorzystania już wytrenowanego modelu lub warstwy w nowym kontekście lub w celu "zamrożenia" pewnych warstw, aby zapobiec ich modyfikacji.
W tym kodzie, ustawiając layer.trainable = False, wyrażamy nasze zamiary wykorzystania wag nauczonego modelu transfer learning, który stosujemy (w tym przypadku InceptionV3) z zestawu danych ImageNet. Ta czynność oznacza konieczność zamrożenia tych wag, aby zapobiec ich modyfikacjom.
include_top=False
Parametr include_top=False jest często używany przy korzystaniu z gotowych modeli architektury sieci neuronowej, takich jak modele z zestawu danych ImageNet w TensorFlow lub Keras.
Gdy include_top=False, warstwa górna sieci neuronowej, która zwykle zawiera w pełni połączone warstwy odpowiedzialne za klasyfikację, jest pomijana. Jest to przydatne, gdy chcemy dostosować gotowy model do własnych potrzeb, na przykład w celu wykorzystania go do transferu uczenia się. W ten sposób możemy zbudować własne warstwy klasyfikacji na podstawie cech wyuczonego modelu, zamiast korzystać z gotowych warstw klasyfikacji, które mogą być specyficzne dla oryginalnego problemu, dla którego był trenowany model.
W skrócie, include_top=False oznacza, że nie będzie używana warstwa górna modelu, co daje nam możliwość dostosowania architektury modelu do naszych własnych potrzeb.


Sequential – budowanie modelu
Sequential grupuje liniowy stos warstw w model.
Zaimplementujemy model sekwencyjny, który będzie zawierać następujące części:
Warstwa Flatten spłaszcza wyjście warstwy konwolucyjnej.
Następnie będziemy mieć dwie w pełni połączone warstwy, po których następuje wyjście z warstwy spłaszczonej.
Ostateczna warstwa to warstwa wyjściowa, która generuje soft probabilities dla dwunastu klas.
Dzięki funkcji tf.nn.softmax mamy rodzaj systemu głosowania dla 10 neuronów w drugiej warstwie. Przyjmuje ona liczby pochodzące z tych neuronów i zamienia je na prawdopodobieństwa. Wyobraź sobie, że masz 10 liczb reprezentujących pewność sieci w różnych opcjach. Te liczby mogą wyglądać tak: [2.0, 3.0, 1.0, 0.1, 2.5, 1.8, 0.5, 1.2, 0.7, 2.2]. Korzystając z tf.nn.softmax, ułatwia się zrozumienie tych liczb. Skompresuje je tak, aby sumowały się do 1. Jest to jak powiedzenie: "Jak pewna jest sieć co do każdej opcji?" Po zastosowaniu tf.nn.softmax liczby mogą wyglądać tak: [0.193, 0.383, 0.043, 0.008, 0.232, 0.137, 0.015, 0.055, 0.025, 0.101]. Teraz te nowe liczby mówią nam o prawdopodobieństwach. Na przykład sieć jest najbardziej pewna (38.3%) drugiej opcji (3.0), a niezbyt pewna (0.8%) czwartej opcji (0.1).

Podczas kompilacji modelu dostarczamy te trzy istotne parametry:

optimizer – Jest to metoda, która pomaga zoptymalizować funkcję kosztu(oznacza funkcję której minimalizacja prowadzi do rozwiązania interesującego nas zadania), korzystając z metody gradientu prostego (algorytm numeryczny mający na celu znalezienie minimum lokalnego zadanej funkcji celu). Wybrano ‘Adam’ - Optymalizacja Adam to metoda stochastycznego spadku gradientowego oparta na adaptacyjnej estymacji momentów pierwszego i drugiego rzędu.
Według Kingma i in., 2014, metoda ta jest "wydajna obliczeniowo, wymaga niewielkiej ilości pamięci, jest niezmienna względem diagonalnego przeskalowania gradientów i dobrze nadaje się do problemów, które są duże pod względem danych/parametrów".
loss – Funkcja straty, za pomocą której monitorujemy, czy model poprawia się w trakcie trenowania czy nie.
sparse_categorical_crossentropy (scce) produkuje indeks kategorii najbardziej prawdopodobnie pasującej kategorii.
Istnieje kilka sytuacji, w których warto użyć sparse_categorical_crossentropy (scce), w tym:
•	gdy twoje klasy są wzajemnie wykluczające się, tj. w ogóle nie obchodzi cię inne dostatecznie bliskie przewidywania,
•	gdy liczba kategorii jest duża, co sprawia, że wynik predykcji staje się przytłaczający.

metrics – Pomaga ocenić model poprzez przewidywanie danych treningowych i walidacyjnych.
Wybrano ‘Accuracy’ - ta metryka tworzy dwie zmienne lokalne, total i count, które są używane do obliczenia częstotliwości, z jaką y_pred zgadza się z y_true. Ta częstotliwość ostatecznie jest zwracana jako dokładność binarna: operacja idempotentna, która po prostu dzieli total przez count.


keras.callbacks.callbacks.EarlyStopping()

Funkcja zwrotna EarlyStopping może monitorować zarówno wartości straty, jak i dokładności. Jeśli zauważalna jest strata, uczenie zostaje zatrzymane, gdy obserwuje się wzrost wartości straty. Natomiast jeśli patrzymy na wskaźniki dokładności (ang. accuracy), to uczenie modelu (ang. training)  zostaje zatrzymane, gdy obserwuje się spadek wartości dokładności (ang. accuracy).

keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', baseline=None, restore_best_weights=False)
gdzie,

monitor: Wartość, którą ma monitorować funkcja, powinna być przypisana. Może to być strata walidacji lub dokładność walidacji.
mode: To tryb, w jakim zmiana w monitorowanej ilości powinna być obserwowana. Może to być 'min' lub 'max' lub 'auto'. Gdy monitorowana wartość to strata, jej wartość to 'min'. Gdy monitorowana wartość to dokładność, jej wartość to 'max'. Gdy tryb jest ustawiony na 'auto', funkcja automatycznie monitoruje z odpowiednim trybem.
min_delta: Minimalna wartość, która powinna być ustawiona dla zmiany, która ma być uwzględniana, czyli zmiana w monitorowanej wartości powinna być wyższa niż wartość 'min_delta'.
patience: Cierpliwość to liczba epok, w których trenowanie ma być kontynuowane po pierwszym zatrzymaniu. Model czeka przez 'patience' liczbę epok na jakiekolwiek ulepszenie w modelu.
verbose: Verbose to wartość całkowita: 0, 1 lub 2. Ta wartość służy do wyboru sposobu wyświetlania postępu podczas trenowania.
Verbose = 0: Tryb cichy - w tym trybie nic nie jest wyświetlane.
Verbose = 1: Wyświetlany jest pasek przedstawiający postęp trenowania.
Verbose = 2: W tym trybie wyświetlany jest jeden wiersz na epokę, pokazujący postęp trenowania na epokę.
restore_best_weights: To wartość logiczna. Wartość True przywraca optymalne wagi.

Liczba epok

Jednym z kluczowych problemów podczas trenowania sieci neuronowej na próbkowych danych jest nadmierne dopasowanie (albo przeuczenie, ang. overfitting). Gdy liczba epok użytych do trenowania modelu sieci neuronowej jest większa niż konieczna, model trenujący uczy się wzorców, które są specyficzne dla danych próbkowych w dużym stopniu. Sprawia to, że model jest niezdolny do skutecznego działania na nowym zestawie danych. Ten model daje wysoką dokładność na zestawie treningowym (danych próbkowych), ale nie osiąga dobrej dokładności na zestawie testowym. Innymi słowy, model traci zdolność generalizacji poprzez dopasowanie danych treningowych. Aby złagodzić nadmierne dopasowanie i zwiększyć zdolność generalizacji sieci neuronowej, model powinien być trenowany przez optymalną liczbę epok. Część danych treningowych jest dedykowana do walidacji modelu, aby sprawdzić wydajność modelu po każdej epoce treningu. Strata i dokładność na zestawie treningowym oraz na zestawie walidacyjnym są monitorowane, aby sprawdzić liczbę epok, po których model zaczyna nadmiernie dopasowywać się.
Uwaga: Trenowanie zostało zatrzymane po 14. epoce, czyli model zacznie nadmiernie dopasowywać się od 15. epoki. W miarę zwiększania liczby epok po 14. wartość straty na zbiorze treningowym maleje i staje się niemal zerowa. Natomiast wartość straty na zbiorze walidacyjnym wzrasta, co pokazuje nadmierne dopasowanie modelu do danych treningowych.

Train & Validation Loss
"Train accuracy" (dokładność treningowa) to metryka mierząca procentowy odsetek poprawnie sklasyfikowanych przykładów w zbiorze treningowym przez model klasyfikacyjny. Jest to ważna metryka oceniająca, jak dobrze model radzi sobie z danymi treningowymi. Jednak sama wysoka dokładność treningowa nie zawsze jest wystarczającym wskaźnikiem jakości modelu. Model może nauczyć się "na pamięć" dane treningowe, co prowadzi do nadmiernego dopasowania (overfitting), co oznacza, że nie generalizuje dobrze do nowych danych. Dlatego ważne jest monitorowanie zarówno dokładności treningowej, jak i walidacyjnej podczas trenowania modelu, aby uniknąć nadmiernego dopasowania.
"Train loss" (strata treningowa) to miara, która określa, jak dobrze model radzi sobie z dopasowaniem się do danych treningowych podczas procesu uczenia. Strata treningowa reprezentuje średnią wartość funkcji straty dla wszystkich przykładów w zbiorze treningowym. Im niższa wartość straty treningowej, tym lepiej model jest w stanie przewidywać etykiety dla danych treningowych. Jednak sam niski poziom straty treningowej niekoniecznie oznacza, że model jest dobry, ponieważ może to prowadzić do nadmiernego dopasowania (overfittingu) modelu do danych treningowych. Dlatego ważne jest równoważenie między minimalizacją straty treningowej a zapobieganiem nadmiernemu dopasowaniu poprzez monitorowanie zarówno straty treningowej, jak i straty walidacyjnej podczas trenowania modelu.


W książkach i tutorialach online często widuje się wykresy, które wyraźnie wskazują, kiedy należy zatrzymać trening. Na przykład, kiedy strata treningowa staje się mniejsza niż strata walidacyjna, lub gdy osiągnięty zostaje plateau w stracie walidacyjnej. Jednakże, w praktyce nie zawsze jest to takie oczywiste, dlatego prosiłbym bardziej doświadczonych użytkowników sieci neuronowych o poradę:
"Validation loss" (strata walidacyjna) to miara, która określa, jak dobrze model radzi sobie z generalizacją do nowych danych, które nie były używane podczas treningu. Strata walidacyjna reprezentuje średnią wartość funkcji straty dla wszystkich przykładów w zbiorze walidacyjnym. Im niższa wartość straty walidacyjnej, tym lepiej model generalizuje do danych, które nie były widziane podczas treningu.
Monitorowanie straty walidacyjnej jest istotne, ponieważ pozwala uniknąć nadmiernego dopasowania (overfittingu) modelu do danych treningowych. Jeśli strata walidacyjna zaczyna rosnąć lub utrzymuje się na stałym poziomie, podczas gdy strata treningowa nadal maleje, może to oznaczać, że model zaczyna się nadmiernie dopasowywać do danych treningowych, a nie generalizuje dobrze do nowych danych.

Dlatego ważne jest monitorowanie zarówno straty treningowej, jak i walidacyjnej podczas trenowania modelu, aby zapewnić, że model będzie dobrze generalizować do nowych danych. Jeśli strata walidacyjna przestanie się poprawiać lub zacznie rosnąć, może to być sygnał, że należy przerwać trening modelu lub dostosować hiperparametry, aby uniknąć nadmiernego dopasowania.
•	Monitoruj różne metryki: Oprócz monitorowania straty treningowej i walidacyjnej, warto obserwować inne metryki, takie jak dokładność czy F1-score. Czasami poprawa w jednej metryce może nie odzwierciedlać rzeczywistego ulepszenia modelu.
•	Wczesne zatrzymywanie: Implementuj technikę wczesnego zatrzymywania, która kończy trening, gdy wybrana metryka na zbiorze walidacyjnym przestanie się poprawiać lub zacznie się pogarszać.
•	Wizualizuj krzywe uczenia: Rysuj wykresy zmian metryk treningowych i walidacyjnych w zależności od liczby epok. Możesz w ten sposób zauważyć tendencje i ocenić, czy model nadal się uczy, czy też powinno się zakończyć trening.
•	Eksperymentuj z hiperparametrami: Czasami brak wyraźnych kryteriów zatrzymania może wynikać z suboptymalnych ustawień hiperparametrów. Próbuj różnych konfiguracji architektury modelu, wartości współczynnika uczenia, czy technik regularyzacji.
•	Konsultuj się z literaturą: Przeglądaj badania naukowe i publikacje związane z Twoją dziedziną problemu. Możesz znaleźć w nich wskazówki dotyczące optymalnych strategii treningowych i kryteriów zatrzymania.
•	Korzystaj z doświadczenia społeczności: Skonsultuj się z innymi doświadczonymi użytkownikami sieci neuronowych, na przykład na forach internetowych lub w grupach dyskusyjnych. Mogą oni podzielić się swoimi doświadczeniami i udzielić cennych porad.
Tablica pomyłek (ang. confusion matrix)
Wiele wartości prawdziwie pozytywnych (duże accuracy)

	Klasa rzeczywista
	pozytywna	negatywna
Klasa
predykowana	pozytywna	prawdziwie
pozytywna (TP)	fałszywie
pozytywna (FP)
	negatywna	fałszywie
negatywna (FN)	prawdziwie
negatywna (TN)


