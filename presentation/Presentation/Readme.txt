
Wichtig!


1.) Videos:
---------------------------

	Diese Präsentation enthält Bilder und Videos.
	Bilder werden beim Kompilieren ins PDF-Dokument
	eingebunden, Videos dagegen nicht. Sie sollen daher
	beim Zeigen der Präsentation unter dem Pfad zu finden sein,
	der beim Kompilieren angegeben worden ist.

	Dieser lässt sich vor dem Kompileren in der Datei
	"PreambleFiles/paths.tex" einmalig setzen.

	Für die Erstellung von Präsentationen für Mitgabe
	werden Videos gerne in das Verzeichnis, wo die
	Präsentation liegt, dazugepakt. Auf diesem Rechner
	zeigt jedoch der Standard-Video-Pfad auf "../Videos"
	(also eine Ebene höher, als die Präsentation), damit
	sie von mehreren Präsentationen verwendet werden können.

	Um die korrekte Wiedergabe für die mitzugendene Präsentationen
	mit Videos in einem Unterverzeichnis zu ermöglichen sollte daher
	- entweder vor dem Erzeugen der entsprechenden PDF
	  der Video-Pfad auf das Unterverzeichnis "Videos" gesetzt werden
	  (dann funktioniert die Präsi mit den Videos aus dem Unterverzeichnis)
	- oder, falls dies versäumt worden ist, kann die Präsentation selbst
	  in ein Unterverzeichnis verschoben werden, damit der relative Pfad stimmt.


2.) Wichtige Hinweise zum Kompilieren:
--------------------------------------


  2.1) Dateien:
  -------------


	Um diese Präsentation zu kompilieren, müssen folgende Dateien
	vorhanden sein:

	- Hauptdatei	(z.B. PresentationSeminarBAF.tex)
	- KITdefs.sty	(Definitionen zu Größen etc. Hier werden auch
				 Pfade zu den Logos definiert)
		
	- Mehrere Einstellungsdateien im Verzeichnis "PreambleFiles":
		- paths.tex			(Einstellungen mit Video- und Bilder-Pfaden)
		- multilang.tex			(Hilfsdatei für Zweisprachigkeit)
		- beamerthemeKIT.sty		(Definitionen zum KIT-Konformen Aussehen der Folien)
		- beameroutherthemeXXXX.tex	(Wichtige Datei, in der die "outer theme"
						 gewählt wird. 
						 Entspricht dem "Master" bei PowerPoint.
						 Hier wird Gestaltung der Titelfolie sowie 
						 Verwendung und Platzierung von Logos, etc.
						 definiert.)
			Je nach Einstellung in der Hauptdatei wird eine der folgenden Dateien verwendet:
		   	- beamerouterthemeIOSB.sty
		   	- beamerouterthemeIOSB-KIT.sty
		   	- beamerouterthemeKIT-IOSB.sty
			- beamerouterthemeKIT-IES.sty
			- beamerouterthemeIES-KIT.sty
		   	- beamerouterthemeIOSB-IES.sty
			- beamerouterthemeIES-IOSB.sty

		- Bilder im Verzeichnis, das in der Datei paths.tex definiert ist
		- Videos braucht man zum Kompileiern nicht,
		  jedoch sollte der Video-Pfad für die spätere Anzeige
		  korrekt gesetzt sein (s.o.)


  2.2) Zweite Sprache (Wechsel Englisch/Deutsch):
  -----------------------------------------------

	Diese Präsentation ist zweisprachig ausgelegt.

	Zweisprachige Texte werden mit dem Befehl

		\transl{English||Deutsch} eingeben.

	Bei mehrzeiligen Angaben auf Kommentierung der Zeilenumbrüche achten,
	damit keine unerwünschte Leerzeichen entstehen:

		\transl{% Hier ein Kommentar, um Leerzeichen zu verhindern
			English Text||% Hier wieder ein Kommentar
			Deutscher Text% Hier wieder
		}% und hier nochmal.

	Die Sprache lässt sich für die Gesamtpräsentation
	und/oder für ihre Teile
	durch ein Makro in der Hauptdatei wählen:

	\setboolean{UseTranslation}{true}%Deutsch

	\setboolean{UseTranslation}{false}%Englisch

	Um Fehler beim Kompilieren schneller finden zu können, 
	empfiehlt es sich die einzelnen Frames in extra Dateien
	auszulagern und sie dann mittels \input einzubinden.
	Diese befinden sich im Unterverzeichnis "Frames".


  2.3) Bilder:
  ------------

	Um diese Präsentation auf einem anderen Rechner
	_KOMPILIEREN_ zu können, müssen Bilder mitkopiert werden.
	Auf diesem Rechner befinden sie sich im Verzeichnis
	"../Images" (also eine Ebene höher als die Präsentation selbst)
	Auf dem Zeilrechner können sie in ein beliebiges Verzeichnis
	kopiert werden. Der Pfad kann einmalig in der Datei
	"paths.tex" eingestellt werden
	(z.B. auf das Unterverzeichnis "Images").

