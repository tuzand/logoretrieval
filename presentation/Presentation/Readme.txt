
Wichtig!


1.) Videos:
---------------------------

	Diese Pr�sentation enth�lt Bilder und Videos.
	Bilder werden beim Kompilieren ins PDF-Dokument
	eingebunden, Videos dagegen nicht. Sie sollen daher
	beim Zeigen der Pr�sentation unter dem Pfad zu finden sein,
	der beim Kompilieren angegeben worden ist.

	Dieser l�sst sich vor dem Kompileren in der Datei
	"PreambleFiles/paths.tex" einmalig setzen.

	F�r die Erstellung von Pr�sentationen f�r Mitgabe
	werden Videos gerne in das Verzeichnis, wo die
	Pr�sentation liegt, dazugepakt. Auf diesem Rechner
	zeigt jedoch der Standard-Video-Pfad auf "../Videos"
	(also eine Ebene h�her, als die Pr�sentation), damit
	sie von mehreren Pr�sentationen verwendet werden k�nnen.

	Um die korrekte Wiedergabe f�r die mitzugendene Pr�sentationen
	mit Videos in einem Unterverzeichnis zu erm�glichen sollte daher
	- entweder vor dem Erzeugen der entsprechenden PDF
	  der Video-Pfad auf das Unterverzeichnis "Videos" gesetzt werden
	  (dann funktioniert die Pr�si mit den Videos aus dem Unterverzeichnis)
	- oder, falls dies vers�umt worden ist, kann die Pr�sentation selbst
	  in ein Unterverzeichnis verschoben werden, damit der relative Pfad stimmt.


2.) Wichtige Hinweise zum Kompilieren:
--------------------------------------


  2.1) Dateien:
  -------------


	Um diese Pr�sentation zu kompilieren, m�ssen folgende Dateien
	vorhanden sein:

	- Hauptdatei	(z.B. PresentationSeminarBAF.tex)
	- KITdefs.sty	(Definitionen zu Gr��en etc. Hier werden auch
				 Pfade zu den Logos definiert)
		
	- Mehrere Einstellungsdateien im Verzeichnis "PreambleFiles":
		- paths.tex			(Einstellungen mit Video- und Bilder-Pfaden)
		- multilang.tex			(Hilfsdatei f�r Zweisprachigkeit)
		- beamerthemeKIT.sty		(Definitionen zum KIT-Konformen Aussehen der Folien)
		- beameroutherthemeXXXX.tex	(Wichtige Datei, in der die "outer theme"
						 gew�hlt wird. 
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
		  jedoch sollte der Video-Pfad f�r die sp�tere Anzeige
		  korrekt gesetzt sein (s.o.)


  2.2) Zweite Sprache (Wechsel Englisch/Deutsch):
  -----------------------------------------------

	Diese Pr�sentation ist zweisprachig ausgelegt.

	Zweisprachige Texte werden mit dem Befehl

		\transl{English||Deutsch} eingeben.

	Bei mehrzeiligen Angaben auf Kommentierung der Zeilenumbr�che achten,
	damit keine unerw�nschte Leerzeichen entstehen:

		\transl{% Hier ein Kommentar, um Leerzeichen zu verhindern
			English Text||% Hier wieder ein Kommentar
			Deutscher Text% Hier wieder
		}% und hier nochmal.

	Die Sprache l�sst sich f�r die Gesamtpr�sentation
	und/oder f�r ihre Teile
	durch ein Makro in der Hauptdatei w�hlen:

	\setboolean{UseTranslation}{true}%Deutsch

	\setboolean{UseTranslation}{false}%Englisch

	Um Fehler beim Kompilieren schneller finden zu k�nnen, 
	empfiehlt es sich die einzelnen Frames in extra Dateien
	auszulagern und sie dann mittels \input einzubinden.
	Diese befinden sich im Unterverzeichnis "Frames".


  2.3) Bilder:
  ------------

	Um diese Pr�sentation auf einem anderen Rechner
	_KOMPILIEREN_ zu k�nnen, m�ssen Bilder mitkopiert werden.
	Auf diesem Rechner befinden sie sich im Verzeichnis
	"../Images" (also eine Ebene h�her als die Pr�sentation selbst)
	Auf dem Zeilrechner k�nnen sie in ein beliebiges Verzeichnis
	kopiert werden. Der Pfad kann einmalig in der Datei
	"paths.tex" eingestellt werden
	(z.B. auf das Unterverzeichnis "Images").

