# How Model 1 and Model 2 Support the Final RHI Model

The final Road Health Index (RHI) model is the combination of two earlier predictions:

1. Model 1 predicts the future surface condition of the road using IRI and traffic-related features.
2. Model 2 identifies the structural health pattern of the road using FWD measurements and clustering.

These two outputs are then used together to compute the final RHI.

## 1. Role of Model 1 Output

Model 1 gives a surface-condition score based on future roughness. This score represents how bad or good the road surface is expected to become over time.

Why this is helpful for RHI:

- It captures the road's ride quality and smoothness.
- It reflects traffic load impact on the pavement surface.
- It provides a numeric surface deterioration score that can be merged into the final index.

In simple terms, Model 1 answers: "How rough will the road become in the future?"

## 2. Role of Model 2 Output

Model 2 analyzes structural health using FWD sensor data. It groups roads into structural clusters such as Good, Fair, or Poor based on similarity in deflection behavior.

Why this is helpful for RHI:

- It captures the internal strength and load-bearing capacity of the pavement.
- It reveals whether the pavement structure is still strong or already weakened.
- It provides the structural health condition that cannot be fully understood from surface roughness alone.

In simple terms, Model 2 answers: "How strong is the road structure underneath the surface?"

## 3. Why Both Models Are Needed

A road can have:

- a smooth surface but weak structural support, or
- a rough surface but strong underlying structure.

So, using only one model would give an incomplete picture. Model 1 captures surface deterioration, while Model 2 captures structural condition. Together, they provide a more complete view of road health.

## 4. How the Third Model Uses Their Outputs

The third model, which calculates RHI, combines the outputs from Model 1 and Model 2 into one final score.

### Standard case

If both Model 1 and Model 2 outputs are available:

- RHI = 50% of the Model 1 score + 50% of the Model 2 score

This balanced approach ensures that both surface condition and structural integrity are considered equally.

### Fallback case

If FWD structural data is missing, the system uses the Model 1 result alone:

- RHI = 100% of the Model 1 score

This fallback is important because it allows the system to still produce a useful road health prediction even when complete structural data is unavailable.

## 5. Final Interpretation

The final RHI is helpful because it converts the separate findings of the two earlier models into one practical decision score.

- Model 1 helps understand future surface damage.
- Model 2 helps understand structural weakness.
- The final RHI helps road authorities decide whether maintenance is required now, soon, or later.

This makes the third model the decision-making layer that combines technical prediction with real-world maintenance planning.
