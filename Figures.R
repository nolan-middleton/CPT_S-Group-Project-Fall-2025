########################################################################
# This R script compiles all the data for the plots.                   #
########################################################################

##### SETUP #####
require(ggplot2)
require(ggforce)
require(jsonlite)
require(stringr)
require(rstudioapi)
require(grid)

# Directories
setwd(dirname(getActiveDocumentContext()$path))

if (!dir.exists("Figures")) {
  dir.create("Figures")
}

# Plotting
plotTheme <- theme(
  plot.margin = margin(t = 10, r = 10, b = 10, l = 10, unit = "pt"),
  plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
  panel.background = element_blank(),
  panel.grid = element_blank(),
  axis.line = element_line(linewidth = 1.5, lineend = "square"),
  axis.ticks = element_line(linewidth = 1.5, color = "black"),
  axis.ticks.length = unit(7.5, "pt"),
  axis.title = element_text(face = "bold", size = 14),
  axis.text = element_text(face = "bold", size = 12, color = "black"),
  legend.text = element_text(face = "bold", size = 12, color = "black"),
  legend.position = "inside",
  legend.position.inside = c(0.05,1),
  legend.justification = c("left", "top"),
  legend.direction = "horizontal",
  legend.margin = margin()
)

# Load data
datasets <- unlist(read.delim("datasets.txt", header = FALSE))
names(datasets) <- c()
models <- c("DecisionTree", "kNearestNeighbours", "NaiveBayes",
            "RandomForest", "SupportVectorMachine")

load_model <- function(L) {
  if ("results" %in% names(L)) {
    return(matrix(unlist(L$results),nrow=length(L$results),byrow=TRUE))
  } else {
    R <- list()
    for (i in 1:length(L)) {
      R[[names(L)[i]]] <- load_model(L[[i]])
    }
    return(R)
  }
}

data <- list()
for (dataset in datasets) {
  print(paste0("> ", dataset, "..."))
  data[[dataset]] <- list()
  for (i in 1:5) {
    print(paste0(">> Strategy ", toString(i), "..."))
    data[[dataset]][[i]] <- list()
    if (i == 1) {
      for (model in models) {
        print(paste0(">>> ", model, "..."))
        data[[dataset]][[i]][[model]] <- load_model(
          read_json(paste0(dataset, "/", toString(i), "/", model, ".json"))
        )
      }
    } else {
      subdirs <- list.dirs(paste0(dataset,"/",toString(i)),full.names=FALSE)
      for (dir in subdirs[2:length(subdirs)]) {
        print(paste0(">>> ", dir, "..."))
        data[[dataset]][[i]][[dir]] <- list()
        for (model in models) {
          data[[dataset]][[i]][[dir]][[model]] <- load_model(
            read_json(
              paste0(dataset,"/",toString(i),"/",dir,"/",model,".json")
            )
          )
        }
      }
    }
  }
}

# Save this object so we don't have to run that loop again
save(data, file = "Figures/dataObj.rda")

# Load the data
data <- load("Figures/dataObj.rda")