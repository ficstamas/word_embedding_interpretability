from src.modules.load.semcor import lexname_as_label
from src.modules.utilities.logging import Logger

Logger().setup("workspace/")

x = lexname_as_label("data/semcor/semcor.data.xml")
print(x)