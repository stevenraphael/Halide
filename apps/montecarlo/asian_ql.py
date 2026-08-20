"""Price the same arithmetic-average Asian call with QuantLib, to validate the
Halide Monte-Carlo pricer. Reads params.txt written by asian_mc."""
import sys, time
import QuantLib as ql

with open("apps/montecarlo/params.txt") as f:
    S0, K, r, sigma, Tyears, T, B, hal_price = f.read().split()
S0, K, r, sigma, Tyears = map(float, (S0, K, r, sigma, Tyears))
T, B = int(T), int(B)
hal_price = float(hal_price)

today = ql.Date(15, 1, 2020)
ql.Settings.instance().evaluationDate = today
dc = ql.Actual365Fixed()
cal = ql.NullCalendar()

spot = ql.SimpleQuote(S0)
rf = ql.FlatForward(today, ql.QuoteHandle(ql.SimpleQuote(r)), dc)
vol = ql.BlackConstantVol(today, cal, ql.QuoteHandle(ql.SimpleQuote(sigma)), dc)
process = ql.BlackScholesProcess(ql.QuoteHandle(spot),
                                 ql.YieldTermStructureHandle(rf),
                                 ql.BlackVolTermStructureHandle(vol))

# T evenly spaced fixing dates over [0, Tyears]; maturity = last fixing.
days = int(round(365 * Tyears))
fixing_dates = [today + int(round(days * (i + 1) / T)) for i in range(T)]
maturity = fixing_dates[-1]

payoff = ql.PlainVanillaPayoff(ql.Option.Call, K)
exercise = ql.EuropeanExercise(maturity)
option = ql.DiscreteAveragingAsianOption(ql.Average.Arithmetic, 0.0, 0,
                                         fixing_dates, payoff, exercise)

# Monte-Carlo engine (matches the Halide methodology).
n_mc = min(B, 200000)
engine = ql.MCDiscreteArithmeticAPEngine(process, "pseudorandom",
                                         requiredSamples=n_mc, seed=42)
option.setPricingEngine(engine)
t0 = time.perf_counter()
mc_price = option.NPV()
mc_ms = (time.perf_counter() - t0) * 1e3
mc_err = option.errorEstimate()

print(f"QuantLib arithmetic Asian call ({T} fixings)")
print(f"  QuantLib MC price = {mc_price:.5f}  +/- {mc_err:.5f}  ({n_mc} paths, {mc_ms:.1f} ms)")
print(f"  Halide MC price   = {hal_price:.5f}")
print(f"  |Halide - QuantLib| = {abs(hal_price - mc_price):.5f}  "
      f"({abs(hal_price - mc_price)/mc_err:.2f} sigma)")
