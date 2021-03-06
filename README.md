# Reviewer-recommendation-by-experience-and-affinity
<pre>
Reviewer recommendation by experience and affinity
This Final Master's Project proposes a solution to the paper reviewer proposal required by the Peer Review process specifically accepted by the academic sector. 
The work of the IJIMAI magazine to propose suitable papers to the collaborators is becoming increasingly complex due to the growing amount of articles they are 
receiving. So far, the task is carried out manually and with this project we intend to make this recommendation effectively and precisely, proposing adjusted 
reviewers in the knowledge area of the article under review. Currently there are many unsupervised recommendation methods and for this project was selected 
the WMD metric tool based on vectors generated by neural networks that allows you to calculate the distance between two documents even without having words in common. 
It uses word embedding generated by Word2Vec and calculates the distance by an adaptation of the calculation of the distance of the earth movement (EMD) to the space 
of the documents. The Word2Vec training is updated with the magazine's papers revised and pending papers this action allows us to addition the new concepts that arise
in the research. The information of the reviewers is updated with the reviews that have obtained the best evaluation by the editor. Finally, together with the result 
of the selected reviewers with the shortest distance to the article and their own information, the calculated attributes are obtained, which are the trust that the 
reviewer shows with the journal, the workload of the last 12 months, if the Reviewer is currently undergoing review and alignment with the publisher and what is the 
style of the majority revisions that he has submitted to the journal. With all this information, the publisher will be able to streamline and improve its reviewer 
assignment process.


1. Fill the file DataEntryTemplate.xlsx

DatosEntradaRevisor sheet:
Usuario revisor -->	Review user
Especialidades -->	Experience and Areas of interest
	comma separated areas of interest
Datos institución y país --> Institution and country information
Autor de la revista	--> Magazine author
	Yes
	No
Miembro del Editorial Board	--> Member of the Editorial Board
	Yes
	No
	Gold
Revisor sugerido por el autor --> Reviewer suggested by the author
	Yes
	No
Título -->	Paper Title
Palabras clave -->	Keywords
	comma separated Keywords
Abstract --> Abstract
Fecha asignación revisión --> Revision assignment date
	Date
Fecha de recepción revisión --> Date of receipt revision
	Date
Versión de la revisión --> Review version
	1a
	2a
	3a
Recomendación revisor --> Reviewer recommendation
	Accept
	Minor Changes
	Mayor Changes
	Reject
Recomendación editor --> Editor recommendation
	Accept
	Minor Changes
	Mayor Changes
	Reject
Estilo revisión --> Review style
	Exhaustive
	Good
	Medium
	Brief
Incidencias --> Incidences
	Not done - Justified
	Not done - Unjustified
DatosArticulos sheet:
Usuario Autor --> Author user
Datos institución y país --> Institution and country information
Fecha recepción del artículo --> Date of receipt paper
Título Artículo --> Paper Title
Palabras clave --> Keywords
	comma separated Keywords
Abstract --> Abstract

2. Execute python tfm.py

3. Example of output:
Distancia --> WMD Distance
Título del artículo --> Article title
Usuario revisor --> Review user
Especialidades --> Experience and Areas of interest
Datos institución y país --> Institution and country information
Autor de la revista --> Magazine author
Miembro del Editorial Board --> Member of the Editorial Board
Confianza --> Trust
Carga de trabajo --> Workload
Revisor en proceso de revisión --> Reviewer in review process
Alineación editorial --> Editorial lineup
Estilo de revisión --> Review style
</pre>
