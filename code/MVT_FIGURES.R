# ============================================================
# MVT PAPER — FINAL PUBLICATION-QUALITY FIGURES (FIXED)
# PNG ONLY — 600 DPI
# ============================================================

## 1. PACKAGES
required_packages <- c("ggplot2","dplyr","tidyr","readr","scales","patchwork")

for (pkg in required_packages) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg, repos = "https://cloud.r-project.org")
  }
}

library(ggplot2)
library(dplyr)
library(tidyr)
library(readr)
library(scales)
library(patchwork)

## 2. WORKING DIRECTORY
setwd("C:/Users/HRIDOY/OneDrive/Desktop/MVT")
dir.create("figures", showWarnings = FALSE)

## 3. GLOBAL SETTINGS
journal_font <- "serif"
ALPHA   <- 0.05
R_SIM   <- 10000

mvt_palette <- c(
  "MVT"="#0072B2", "Bartlett"="#E69F00", "Levene"="#009E73",
  "GQ"="#D55E00",  "White"="#CC79A7",    "BP"="#56B4E9",
  "ICSS"="#666666"
)
test_levels <- names(mvt_palette)

## 4. THEME
theme_mvt <- function(base_size = 11) {
  theme_classic(base_size = base_size, base_family = journal_font) +
    theme(
      plot.title       = element_text(size = 13, face = "bold",
                                      hjust = 0.5, margin = margin(b = 7)),
      plot.subtitle    = element_text(size = 9.5, hjust = 0.5,
                                      colour = "grey25", margin = margin(b = 9)),
      axis.title       = element_text(size = 10.5, face = "bold"),
      axis.text        = element_text(size = 9),
      axis.line        = element_line(linewidth = 0.5),
      axis.ticks       = element_line(linewidth = 0.4),
      legend.title     = element_text(size = 9.5, face = "bold"),
      legend.text      = element_text(size = 8.8),
      legend.position  = "bottom",
      legend.key.width = unit(1.15, "cm"),
      legend.key.height= unit(0.45, "cm"),
      strip.text       = element_text(size = 10, face = "bold"),
      strip.background = element_rect(fill = "grey95",
                                      colour = "grey70",
                                      linewidth = 0.35),
      panel.grid       = element_blank(),
      plot.margin      = margin(8, 10, 8, 10)
    )
}

## ============================================================
## FIGURE 1 — TYPE I ERROR
## ============================================================
type1_file <- "MVT_vs_Classical_Model1_TypeI.csv"
if (!file.exists(type1_file)) stop("File not found: ", type1_file)

type1 <- read_csv(type1_file, show_col_types = FALSE)

type1_long <- type1 %>%
  pivot_longer(cols = all_of(test_levels),
               names_to = "Test", values_to = "Type_I_Error") %>%
  mutate(
    Test = factor(Test, levels = test_levels),
    n    = factor(as.character(.data$n),
                  levels = as.character(sort(unique(.data$n)))),
    k    = factor(.data$k)
  )

mc_se    <- sqrt(ALPHA * (1 - ALPHA) / R_SIM)
mc_lower <- ALPHA - 1.96 * mc_se
mc_upper <- ALPHA + 1.96 * mc_se

fig1 <- ggplot(type1_long,
               aes(Test, Type_I_Error, colour = Test, shape = k)) +
  annotate("rect", xmin = -Inf, xmax = Inf,
           ymin = mc_lower, ymax = mc_upper,
           fill = "#D9EAF7", alpha = 0.55, colour = NA) +
  geom_hline(yintercept = ALPHA, linetype = "dashed",
             linewidth = 0.65, colour = "black") +
  geom_point(size = 2.7, stroke = 0.7,
             position = position_jitter(width = 0.09, height = 0, seed = 12345)) +
  facet_wrap(~ n, ncol = 2) +
  scale_colour_manual(values = mvt_palette) +
  scale_y_continuous(limits = c(0.03, 0.065),
                     breaks = seq(0.03, 0.065, by = 0.005),
                     labels = number_format(accuracy = 0.001)) +
  labs(title = "Type I Error Calibration",
       subtitle = "Empirical rejection rates under the homoscedastic null",
       x = "Variance-stability test",
       y = "Empirical Type I error",
       colour = "Test", shape = expression(k)) +
  theme_mvt(10.5) +
  theme(axis.text.x = element_text(angle = 30, hjust = 1))

ggsave("figures/Figure_1_TypeI_Error_Calibration.png",
       fig1, width = 7.2, height = 6.0, units = "in",
       dpi = 600, bg = "white", type = "cairo-png")

## ============================================================
## FIGURE 2 — POWER
## ============================================================
power_file <- "MVT_vs_Classical_Model2_Power.csv"
if (!file.exists(power_file)) stop("File not found: ", power_file)

power <- read_csv(power_file, show_col_types = FALSE)
req_cols <- c("n","k","eta","sigma2_sq", test_levels)
miss <- setdiff(req_cols, names(power))
if (length(miss)) stop("Missing columns: ", paste(miss, collapse = ", "))

power_fig <- power %>%
  filter(n == 500, k == 5, abs(eta - 0.50) < 1e-8) %>%
  pivot_longer(all_of(test_levels),
               names_to = "Test", values_to = "Power") %>%
  mutate(Test = factor(Test, levels = test_levels))

if (!nrow(power_fig)) stop("power_fig empty — check filter values.")

fig2 <- ggplot(power_fig,
               aes(sigma2_sq, Power,
                   colour = Test, linetype = Test,
                   shape = Test, group = Test)) +
  geom_line(linewidth = 0.75) +
  geom_point(size = 2.3, stroke = 0.6) +
  scale_colour_manual(values = mvt_palette) +
  scale_x_continuous(breaks = c(1.10,1.25,1.50,2,3,5),
                     labels = c("1.10","1.25","1.50","2.00","3.00","5.00")) +
  scale_y_continuous(limits = c(0,1.02),
                     breaks = seq(0,1,by=0.2)) +
  labs(title = "Power under Variance Changes",
       subtitle = "Representative: n = 500, k = 5, eta = 0.50",
       x = expression("Post-break variance (" * sigma[2]^2 * ")"),
       y = "Empirical power") +
  theme_mvt(10.5)

ggsave("figures/Figure_2_Power_Comparison.png",
       fig2, width = 7.2, height = 5.4, units = "in",
       dpi = 600, bg = "white", type = "cairo-png")

## ============================================================
## FIGURE 3 — GOLD PRICE & RETURNS
## ============================================================
gold_file <- "Gold Futures Historical Data.csv"
if (!file.exists(gold_file)) stop("File not found: ", gold_file)

gold_raw <- read_csv(gold_file, show_col_types = FALSE)

price_candidates <- c("Price","Close","Last")
price_col <- intersect(price_candidates, names(gold_raw))[1]
if (is.na(price_col)) stop("No price column found.")

gold <- gold_raw %>%
  mutate(
    Date       = as.Date(.data$Date),
    PriceValue = as.numeric(gsub("[,\\s]", "", as.character(.data[[price_col]])))
  ) %>%
  filter(!is.na(Date), !is.na(PriceValue), PriceValue > 0) %>%
  arrange(Date) %>%
  mutate(Log_Return = c(NA_real_, diff(log(PriceValue))))

if (nrow(gold) < 10) stop("Too few valid gold observations.")

p_price <- ggplot(gold, aes(Date, PriceValue)) +
  geom_line(linewidth = 0.42, colour = "#0072B2") +
  scale_x_date(date_breaks = "3 years", date_labels = "%Y") +
  scale_y_continuous(labels = comma) +
  labs(x = NULL, y = "Gold futures price") +
  theme_mvt(10)

p_return <- ggplot(gold, aes(Date, Log_Return)) +
  geom_line(linewidth = 0.32, colour = "#333333") +
  geom_hline(yintercept = 0, linetype = "dashed",
             linewidth = 0.45, colour = "#D55E00") +
  scale_x_date(date_breaks = "3 years", date_labels = "%Y") +
  labs(x = "Date", y = "Log return") +
  theme_mvt(10)

fig3 <- (p_price / p_return) +
  plot_annotation(
    title = "Gold Futures Prices and Log Returns",
    theme = theme(plot.title = element_text(family = journal_font,
                                            size = 13, face = "bold",
                                            hjust = 0.5))
  )

ggsave("figures/Figure_3_Gold_Price_and_Log_Returns.png",
       fig3, width = 7.2, height = 7.0, units = "in",
       dpi = 600, bg = "white", type = "cairo-png")

## ============================================================
## FIGURE 4 — MVT k-SENSITIVITY
## ============================================================
sens_file <- "Gold_MVT_k_sensitivity.csv"
if (!file.exists(sens_file)) stop("File not found: ", sens_file)

sensitivity <- read_csv(sens_file, show_col_types = FALSE)

p_col <- intersect(c("p_value","p.value","pvalue","P_Value","p"),
                   names(sensitivity))[1]
k_col <- intersect(c("k","K","segments","Segments"),
                   names(sensitivity))[1]
if (is.na(p_col) || is.na(k_col))
  stop("Missing p or k column. Available: ",
       paste(names(sensitivity), collapse = ", "))

sensitivity <- sensitivity %>%
  rename(K_Value = all_of(k_col), P_Value = all_of(p_col)) %>%
  mutate(
    P_Value    = pmax(P_Value, .Machine$double.xmin),
    MinusLog10P = -log10(P_Value)
  )

k_breaks <- if (length(sensitivity$K_Value) <= 10)
  sensitivity$K_Value else pretty(range(sensitivity$K_Value), n = 8)

fig4 <- ggplot(sensitivity, aes(K_Value, MinusLog10P)) +
  geom_line(linewidth = 0.70, colour = "#0072B2") +
  geom_point(size = 2.1, stroke = 0.6, colour = "#0072B2") +
  geom_hline(yintercept = -log10(ALPHA), linetype = "dashed",
             linewidth = 0.65, colour = "#D55E00") +
  scale_x_continuous(breaks = k_breaks) +
  scale_y_continuous(labels = number_format(accuracy = 1)) +
  labs(title = "Sensitivity of the MVT to the Number of Segments",
       subtitle = "Gold futures log returns; dashed line: alpha = 0.05",
       x = expression("Number of segments (" * k * ")"),
       y = expression(-log[10] * "(MVT p-value)")) +
  theme_mvt(10.5)

ggsave("figures/Figure_4_MVT_k_Sensitivity.png",
       fig4, width = 7.2, height = 5.2, units = "in",
       dpi = 600, bg = "white", type = "cairo-png")

## ============================================================
## FIGURE 5 — GOLD SEGMENT VARIANCES
## ============================================================
seg_file <- "Gold_MVT_Segment_Variances.csv"
if (!file.exists(seg_file)) stop("File not found: ", seg_file)

segment <- read_csv(seg_file, show_col_types = FALSE)

seg_col <- intersect(c("Segment","segment","j","Index"),
                     names(segment))[1]
var_col <- intersect(c("Variance","variance","Segment_Variance","v"),
                     names(segment))[1]
if (is.na(seg_col) || is.na(var_col))
  stop("Missing segment/variance column. Available: ",
       paste(names(segment), collapse = ", "))

segment <- segment %>%
  rename(Segment_ID = all_of(seg_col),
         Segment_Variance = all_of(var_col))

mean_variance <- mean(segment$Segment_Variance, na.rm = TRUE)
x_max <- max(segment$Segment_ID, na.rm = TRUE)

fig5 <- ggplot(segment, aes(Segment_ID, Segment_Variance)) +
  geom_line(linewidth = 0.55, colour = "#0072B2") +
  geom_point(size = 1.65, stroke = 0.5, colour = "#0072B2") +
  geom_hline(yintercept = mean_variance, linetype = "dashed",
             linewidth = 0.65, colour = "#D55E00") +
  scale_x_continuous(breaks = seq(0, x_max, by = 10),
                     limits = c(0, x_max)) +
  scale_y_continuous(labels = number_format(accuracy = 0.1)) +
  labs(title = "Segment-wise Variance Estimates",
       subtitle = "Gold futures log returns with k = 99 segments",
       x = "Segment",
       y = expression("Segment variance (" * v[j] * ")")) +
  theme_mvt(10.5)

ggsave("figures/Figure_5_Gold_Segment_Variances.png",
       fig5, width = 7.2, height = 5.2, units = "in",
       dpi = 600, bg = "white", type = "cairo-png")

## ============================================================
## FINAL CHECK
## ============================================================
cat("\n====================================================\n")
cat("MVT FIGURES GENERATED SUCCESSFULLY\n")
cat("====================================================\n\n")
cat("Output directory:\n"); print(normalizePath("figures"))
cat("\nGenerated PNG files:\n\n")
print(list.files("figures", pattern = "\\.png$"))
cat("\n====================================================\n")