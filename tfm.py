
import tkinter
import tkinter.filedialog
import tkinter.messagebox
import tkinter.simpledialog
import tkinter.ttk

import nltk
import nltk.corpus

import json
import gensim.models
import pyemd.emd
import pandas

import datetime
import heapq

import os
import os.path


class App:
	def __init__(self, master):
		self.frame = tkinter.Frame(master)
		self.frame.pack(fill="both", expand="yes")
		self.canvas = tkinter.Canvas(self.frame)
		self.canvas.pack(fill="both", expand="yes")
		self.texty=0

class Configuracion(tkinter.simpledialog.Dialog):
	def body(self, master):
		tkinter.Label(master, text="Fichero de datos orígen").grid(row=0, padx=5, pady=5)
		tkinter.Label(master, text="Fichero de datos resultado").grid(row=1, padx=5, pady=5)
		tkinter.Label(master, text="Número de candidatos a revisar").grid(row=2, padx=5, pady=5)

		self.ficheroOrigen = tkinter.Entry(master, width=64)
		self.ficheroOrigen.insert(0,configuracion["ficheroOrigen"])
		self.ficheroResultado = tkinter.Entry(master, width=64)
		self.ficheroResultado.insert(0,configuracion["ficheroResultado"])
		self.numrevisores = tkinter.Entry(master, width=5)
		self.numrevisores.insert(0,configuracion["numrevisores"])
		
		self.ficheroOrigen.grid(row=0, column=1, padx=5, pady=5)
		self.ficheroResultado.grid(row=1, column=1, padx=5, pady=5)
		self.numrevisores.grid(row=2, sticky="W", column=1, padx=5, pady=5)
		
		self.botonBusquedaficheroOrigen=tkinter.Button(master,text='Buscar...',command=self.BotonBusquedaficheroOrigen)
		self.botonBusquedaficheroOrigen.grid(row=0, column=2, padx=5, pady=5)

		self.botonBusquedaficheroResultado=tkinter.Button(master,text='Buscar...',command=self.BotonBusquedaficheroResultado)
		self.botonBusquedaficheroResultado.grid(row=1, column=2, padx=5, pady=5)
		
		return self.ficheroOrigen # initial focus

	def BotonBusquedaficheroOrigen(self):
		resultado=tkinter.filedialog.askopenfilename(title="Fichero de entrada",filetypes=(("Excel","*.xlsx"),("Todos","*.*")))
		if resultado!='':
			self.ficheroOrigen.delete(0, tkinter.END)
			self.ficheroOrigen.insert(0, resultado)
		
	def BotonBusquedaficheroResultado(self):
		resultado=tkinter.filedialog.asksaveasfilename(title="Salvar fichero predicción",filetypes=(("Excel","*.xlsx"),("Todos","*.*")))
		if resultado!='':
			self.ficheroResultado.delete(0, tkinter.END)
			self.ficheroResultado.insert(0, resultado)
		
	def apply(self):
		configuracion["ficheroOrigen"]=self.ficheroOrigen.get()
		configuracion["ficheroResultado"]=self.ficheroResultado.get()
		try:
			configuracion["numrevisores"]=int(self.numrevisores.get())
		except ValueError:
			configuracion["numrevisores"]=5
		
def menuConfiguracion():
	Configuracion(root)

def menuSalir():
	if tkinter.messagebox.askyesno("Salir", "Cerrar aplicacion"):
		root.destroy()

def acercade():
	tkinter.messagebox.showinfo("Acerca de...", "Recomendador de revisores por experiencia y afinidad para la revista IJIMAI\n2020 Laia Garriga")

def preprocess(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    doc = doc.lower().replace('-',' ')  # Lower the text.
    doc = nltk.word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc

def preprocess_comas(doc):
    stop_words = nltk.corpus.stopwords.words('english')
    doc = doc.lower().replace('-',' ')  # Lower the text.
    doc = nltk.word_tokenize(doc)  # Split into words.		
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    list_actual = []
    list_doc = []
    for w in doc:
        if w==',':
            if len(list_actual)>0:
                list_doc.append(list_actual)
                list_actual=[]
        else:
            if w.isalpha():
                list_actual.append(w)
    if len(list_actual)>0:
        list_doc.append(list_actual)
    return list_doc

def recalculoWord2Vec(force=True):
	global modelW2V
	
	if not(force):
		try:
			modelW2V = gensim.models.Word2Vec.load('modelW2V')
		except FileNotFoundError:
			force=True
		
	if force:
		try:
			ds=pandas.read_excel(configuracion['ficheroOrigen'],sheet_name=['DatosEntradaRevisor','DatosArticulos'],keep_default_na=False)
		except FileNotFoundError:
			tkinter.messagebox.showerror("Error", "Fichero "+configuracion['ficheroOrigen']+" no encontrado")
			return
	
		w2v_corpus = []  # Documents to train word2vec

		for x in ds['DatosEntradaRevisor']['Especialidades']: 
			w2v_corpus+=preprocess_comas(x)
		w2v_corpus+=[ preprocess(x) for x in ds['DatosEntradaRevisor']['Título']]
		for x in ds['DatosEntradaRevisor']['Palabras clave']:
			w2v_corpus+=preprocess_comas(x)			
		w2v_corpus+=[ preprocess(x) for x in ds['DatosEntradaRevisor']['Abstract']]
		w2v_corpus+=[ preprocess(x) for x in ds['DatosArticulos']['Título Artículo']]
		for x in ds['DatosArticulos']['Palabras clave']:
			w2v_corpus+=preprocess_comas(x)		
		w2v_corpus+=[ preprocess(x) for x in ds['DatosArticulos']['Abstract']]
		
		# dataset publico arXiv (http://www.kaggle.com/ open source library for research papers)
		with open('arxivData.json', encoding="utf8") as json_file:
			data = json.load(json_file)
			for p in data:
				# Add to corpus for training Word2Vec.
				w2v_corpus.append(preprocess(p['summary']))

		modelW2V = gensim.models.Word2Vec(w2v_corpus, workers=3, size=100)
		modelW2V.save('modelW2V')
	
	
def menuRecalculoWord2Vec():
	recalculoWord2Vec(True)
	tkinter.messagebox.showinfo("Info", "Recálculo finalizado")
	
def menuGeneracion():
	try:
		try:
			ds=pandas.read_excel(configuracion['ficheroOrigen'],sheet_name=['DatosEntradaRevisor','DatosArticulos'],keep_default_na=False)
		except FileNotFoundError:
			tkinter.messagebox.showerror("Error", "Fichero "+configuracion['ficheroOrigen']+" no encontrado")
			return

		recalculoWord2Vec(False)
		
		resultados=[]
		maxrevisores=configuracion['numrevisores']
	
		for indexart,art in ds['DatosArticulos'].iterrows():
			revisores=[]
			articulo=preprocess(art['Título Artículo'])
			articulo+=preprocess(art['Palabras clave'])
			articulo+=preprocess(art['Abstract'])
			
			for rev in ds['DatosEntradaRevisor']['Usuario revisor'].unique():
				revisor=[]
				autor='No'
				miembro='No'
				fecha_miembro=datetime.datetime.min
				confianza=0
				articulos=0
				carga=0
				ultima_rev_en_curso=datetime.datetime.min
				alineacion=0
				estilo={}       
				
				for indexrev,revdetail in ds['DatosEntradaRevisor'][ds['DatosEntradaRevisor']['Usuario revisor']==rev].iterrows():
					if len(revisor)==0: # Eliminar esta fila si se quiere considerar la especialidad repetidas veces
						revisor+=preprocess(revdetail['Especialidades'])

					if revdetail['Estilo revisión'] in [ 'Exhaustive','Good','Medium' ]:
						revisor+=preprocess(revdetail['Título'])
						revisor+=preprocess(revdetail['Palabras clave'])
						revisor+=preprocess(revdetail['Abstract'])

					if revdetail['Autor de la revista']=='Yes':
						autor=='Yes'
						
					if revdetail['Miembro del Editorial Board']!='' and fecha_miembro>revdetail['Fecha asignación revisión']:
						miembro=revdetail['Miembro del Editorial Board']
						
					if revdetail['Incidencias']=='Not done - Unjustified':
						confianza+=1

					articulos+=1

					if pandas.isnull(revdetail['Fecha de recepción revisión']) and not(pandas.isnull(revdetail['Fecha asignación revisión'])):
						carga+=1

					if pandas.isnull(revdetail['Fecha de recepción revisión']) and not(pandas.isnull(revdetail['Fecha asignación revisión'])) and revdetail['Fecha asignación revisión']>ultima_rev_en_curso:
						ultima_rev_en_curso=revdetail['Fecha asignación revisión']

					if not(pandas.isnull(revdetail['Recomendación revisor'])) and revdetail['Recomendación revisor']==revdetail['Recomendación editor']:
						alineacion+=1

					if revdetail['Estilo revisión']!='':
						estilo[revdetail['Estilo revisión']]=estilo.get(revdetail['Estilo revisión'],0)+1
						
				confianza=(1.0-confianza/articulos)*100
				if len(estilo)==0:
					estilo=''
				else:
					estilo=max(estilo, key=estilo.get)
				
				alineacion=(alineacion/articulos)*100
				
				if ultima_rev_en_curso==datetime.datetime.min:
					ultima_rev_en_curso=''
					
				d=modelW2V.wv.wmdistance(articulo,revisor)
							
				if len(revisores)==maxrevisores:
					if d<-revisores[0][0]:
						heapq.heappushpop(revisores,(-d,art['Título Artículo'],rev,revdetail['Especialidades'],
						revdetail['Datos institución y país'],autor,miembro,confianza,carga,ultima_rev_en_curso,alineacion,estilo))
				else:
					heapq.heappush(revisores,(-d,art['Título Artículo'],rev,revdetail['Especialidades'],
						revdetail['Datos institución y país'],autor,miembro,confianza,carga,ultima_rev_en_curso,alineacion,estilo))
				
			revisores.sort(reverse=True)
			resultados+=revisores
	except:
		tkinter.messagebox.showerror("Error", "Formato de fichero de entrada incorrecto, revisar en el apartado de configuración si se ha seleccionado el adecuado.")
		return

	try:	
		resultado=pandas.DataFrame(data=resultados,columns=['Distancia','Título del artículo','Usuario revisor','Especialidades','Datos institución y país',
											'Autor de la revista','Miembro del Editorial Board','Confianza','Carga de trabajo',
											'Revisor en proceso de revisión','Alineación editorial','Estilo de revisión'])

		resultado['Distancia']=-resultado['Distancia']
		resultado.to_excel(configuracion['ficheroResultado'],sheet_name='Resultados',index=False)
			
		tkinter.messagebox.showinfo("Info", "Generación finalizada")
	except:
		tkinter.messagebox.showerror("Error", "No se ha podido escribir en el fichero de resultado, revisar si esta abierto o si la ubicación en el apartado de configuración es incorrecto o si no has informado la extensión .xlsx.")



root = tkinter.Tk()
root.geometry("600x400")
root.title('Recomendador Revisor')
app = App(root)

statusbar = tkinter.Label(root, text="", bd=1, relief=tkinter.SUNKEN, anchor=tkinter.W)
statusbar.pack(side=tkinter.BOTTOM, fill=tkinter.X)

menu = tkinter.Menu(root)
root.config(menu=menu)

filemenu = tkinter.Menu(menu)
menu.add_cascade(label="Proceso", menu=filemenu)
filemenu.add_command(label="Generación fichero recomendaciones...", command=menuGeneracion)
filemenu.add_command(label="Recálculo Word2Vec...", command=menuRecalculoWord2Vec)
filemenu.add_separator()
filemenu.add_command(label="Salir", command=menuSalir)

configmenu = tkinter.Menu(menu)
menu.add_cascade(label="Configuración", menu=configmenu)
configmenu.add_command(label="Configuración", command=menuConfiguracion)

helpmenu = tkinter.Menu(menu)
menu.add_cascade(label="Ayuda", menu=helpmenu)
helpmenu.add_command(label="Acerca de...", command=acercade)

configuracion={"ficheroOrigen":"FormatoDatosEntrada.xlsx","ficheroResultado":"resultados.xlsx","numrevisores":5}
		
try:
	with open('config') as infile:
		configuracion = json.load(infile)
except FileNotFoundError:
	pass
except:
	print("Fichero de configuracion incorrecto")

nltk.download('stopwords')

root.mainloop()

with open('config', 'w') as outfile:
    json.dump(configuracion, outfile)
