import win32com.client
import numpy as np

class IOMRFileHandler:
    """
    IOptical Measurement Result (OMR) File Handler
    """
    def __init__(self):
        self._engine = win32com.client.Dispatch('AgServerOMRFileHandler.OMRFile')

    @property
    def engine(self):
        return self._engine
    
    @engine.setter
    def engine(self, engine):
        self._engine = engine

    @property
    def figure(self):
        return self._figure
    
    @figure.setter
    def figure(self, figure):
        self._figure = figure

    # IOMR File Handler
    def properties(self, name: str):
        return self.engine.Property(name)
    
    def graph_names(self):
        return self.engine.GraphNames
    
    def graph(self, name: str):
        return self.engine.Graph(name)
    
    def plugins(self):
        return self.engine.Plugins
    
    def plugin(self, name: str):
        self._plugin = self.engine.Plugin(name)
        return self._plugin
    
    def write_omr(self, filename: str):
        self.engine.Write(filename)

    def read_omr(self, filename: str):
        self.engine.OpenRead(filename)

    def close(self):
        self.engine.Close()

    # IOMR Property Handler
    def is_file_compatible(self, graph: str):
        return self._plugin.IsFileCompatible(graph)
    
    def plugin_evaluate(self, name: str):
        return self._plugin.Evaluate(name)
    
    @property
    def settings_xml(self):
        return self._plugin.SettingsXML
    
    @settings_xml.setter
    def settings_xml(self, value: str):
        self._plugin.SettingsXML = value


class IOMRGraphHandler:
    # IOMR Graph Handler
    def __init__(self, graph):
        self.graph = graph

    def properties(self, name: str):
        return self.graph.Property(name)
    
    def process_xdata(self):
        xdata = self.graph.XData
        if not xdata:
            xdata = np.round(np.arange(self.xstart, self.xstop + self.xstep, self.xstep), 14)
        return xdata
    
    @property
    def xdata(self):
        return self.process_xdata()
    
    @property
    def ydata(self):
        return self.graph.YData
    
    @property
    def xstart(self):
        return self.graph.xStart
    
    @property
    def xstop(self):
        return self.graph.xStop
    
    @property
    def xstep(self):
        return self.graph.xStep
    
    @property
    def noChannels(self):
        return self.graph.noChannels
    
    @property
    def noCurves(self):
        return self.graph.noCurves
    
    @property
    def dataPerCurve(self):
        return self.graph.dataPerCurve
    

class IOMRPropertyHandler:
    def __init__(self, pty):
        self._property = pty

    # IOMR Property Handler
    def property_names(self):
        return self._property.PropertyNames
    
    def properties(self, name: str):
        return self._property.Property(name)

    @property
    def value(self, name: str):
        return self._property[name].Value
    
    @value.setter
    def value(self, name: str, value: float):
        self._property[name].Value = value

    @property
    def flag_info_pane(self, name: str):
        return self._property[name].FlagInfoPane
    
    @flag_info_pane.setter
    def flag_info_pane(self, name: str, value: bool):
        self._property[name].FlagInfoPane = value

    @property
    def flag_hide(self, name: str):
        return self._property[name].FlagHide
    
    @flag_hide.setter
    def flag_hide(self, name: str, value: bool):
        self._property[name].FlagHide = value
