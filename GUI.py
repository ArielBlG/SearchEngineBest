import PySimpleGUI as sg
sg.theme('Light Green 2')

class GUI:
    def __init__(self):

        self.file_list_column = [[sg.Text("Enter a Query:"), sg.In(size=(50, 1), key="-QUERY-"),
                                  sg.Button("Search", key="-SEARCH-"), ],
                                 ]
        self.image_viewer_column = [[sg.Text("Pick a Document \nfrom the list below:")
                                   , sg.Listbox(values=[], enable_events=True, size=(30, 10), key="-ANSWERS-")], ]
        self.doc_viewer = [[sg.Text("Tweet full text:"),sg.Output(key="-DOCTEXT-", size=(80, 10))]]

        self.layout = [
            [
                sg.Column([[sg.Text("Loading please wait...")]], key='-LOADING-'),
                sg.Column(self.file_list_column, visible=False, key="-COL1-"),
                sg.Column(self.image_viewer_column, visible=False, key='-COL2-'),
                sg.Column(self.doc_viewer, visible=False, key='-COL3-')]
        ]
        self.window = sg.Window("Tweeter Search Engine", self.layout)

    def runGUI(self):
        event, values = self.window.read()
        if event == '-SEARCH-':
            self.window["-DOCTEXT-"].update(value="")
            return 1, values["-QUERY-"]
        if event == "Exit" or event == sg.WIN_CLOSED:
            return 1, 'Exit'
        if event == "-ANSWERS-":
            return 2, values["-ANSWERS-"]
        return None

    def closeWindow(self):
        self.window.close()

    def switchWindow(self):
        self.window["-LOADING-"].update(visible=False)
        self.window["-COL1-"].update(visible=True)
        self.window["-COL2-"].update(visible=True)
        self.window["-COL3-"].update(visible=True)
        self.window["-COL3-"].update(visible=True)
        self.window["-DOCTEXT-"].update(value="")

    def updateAnswers(self, answers):
        self.window["-ANSWERS-"].update(answers)

    def setDocText(self, text):
        self.window["-DOCTEXT-"].update(value=text)