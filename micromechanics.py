from dataclasses import dataclass, field


def _vf(v: float) -> None:
    if not 0 < v < 1:
        raise ValueError(f"vf must be between 0 and 1 (got {v})")


@dataclass
class Material:
    E: float
    nu: float
    G: float | None = None
    rho: float | None = None
    name: str = "material"
    K: float = field(init=False)

    def __post_init__(self) -> None:
        if self.E <= 0 or self.G is not None and self.G <= 0:
            raise ValueError(f"{self.name}: moduli must be positive")
        if not -1 < self.nu < .5:
            raise ValueError(f"{self.name}: nu must be between -1 and 0.5")
        if self.rho is not None and self.rho <= 0:
            raise ValueError(f"{self.name}: rho must be positive")
        self.G = self.G or self.E / (2 * (1 + self.nu))
        self.K = self.E / (3 * (1 - 2 * self.nu))


def rule_of_mixtures(f: float, m: float, vf: float) -> float:
    _vf(vf)
    return vf * f + (1 - vf) * m


def inverse_rule_of_mixtures(f: float, m: float, vf: float) -> float:
    _vf(vf)
    return 1 / (vf / f + (1 - vf) / m)


def halpin_tsai(f: float, m: float, vf: float, xi: float) -> float:
    _vf(vf)
    eta = (f / m - 1) / (f / m + xi)
    return m * (1 + xi * eta * vf) / (1 - eta * vf)


@dataclass
class Result:
    vf: float
    method_E2: str
    method_G12: str
    E1: float
    E2: float
    nu12: float
    nu21: float
    nu23: float
    G12: float
    G23: float
    K: float
    rho: float | None = None

    E3 = property(lambda s: s.E2)
    nu13 = property(lambda s: s.nu12)
    nu31 = property(lambda s: s.nu21)
    nu32 = property(lambda s: s.nu23)
    G13 = property(lambda s: s.G12)

    def as_dict(self) -> dict:
        values = {"vf": self.vf, "E1": self.E1, "E2=E3": self.E2,
                  "nu12=nu13": self.nu12, "nu21=nu31": self.nu21,
                  "nu23=nu32": self.nu23, "G12=G13": self.G12,
                  "G23": self.G23, "K": self.K,
                  "method_E2": self.method_E2, "method_G12": self.method_G12}
        if self.rho is not None:
            values["rho"] = self.rho
        return values

    def compliance_3D(self): return compliance_matrix_3D(self)
    def compliance_2D(self): return compliance_matrix_2D(self)
    def stiffness_2D(self): return reduced_stiffness_2D(self)

    def print_report(self, unit: str = "GPa") -> None:
        rows = [("E1 (longitudinal)", self.E1, unit),
                ("E2 = E3 (transverse)", self.E2, unit),
                ("G12 = G13 (in-plane shear)", self.G12, unit),
                ("nu12 = nu13 (major)", self.nu12, ""),
                ("nu23 = nu32 (transverse)", self.nu23, ""),
                ("nu21 = nu31 (minor)", self.nu21, ""),
                ("G23 (transverse shear)", self.G23, unit),
                ("K (bulk modulus)", self.K, unit)]
        if self.rho is not None:
            rows.append(("rho (density)", self.rho, ""))
        print(f"\nEffective ply constants (vf = {self.vf:.3f})")
        print(f"E2: {self.method_E2}; G12: {self.method_G12}")
        _print_values(rows)

    def print_matrices(self, unit: str = "GPa") -> None:
        _print_matrix(self.compliance_3D(), f"S (3D compliance) [1/{unit}]",
                      ["1", "2", "3", "4=23", "5=13", "6=12"])
        _print_matrix(self.compliance_2D(), f"S (plane stress) [1/{unit}]",
                      ["1", "2", "6=12"])
        _print_matrix(self.stiffness_2D(), f"Q (plane stress) [{unit}]",
                      ["1", "2", "6=12"])


def compute(fibre: Material, matrix: Material, vf: float, *, xi_E2: float = 2,
            xi_G12: float = 1, use_halpin_tsai: bool = True) -> Result:
    _vf(vf)
    method = lambda xi: f"Halpin-Tsai (xi={xi:g})" if use_halpin_tsai else "Reuss"
    E1 = rule_of_mixtures(fibre.E, matrix.E, vf)
    E2 = (halpin_tsai(fibre.E, matrix.E, vf, xi_E2) if use_halpin_tsai
          else inverse_rule_of_mixtures(fibre.E, matrix.E, vf))
    nu12 = rule_of_mixtures(fibre.nu, matrix.nu, vf)
    nu21 = nu12 * E2 / E1
    K = inverse_rule_of_mixtures(fibre.K, matrix.K, vf)
    nu23 = 1 - nu21 - E2 / (3 * K)
    G12 = (halpin_tsai(fibre.G, matrix.G, vf, xi_G12) if use_halpin_tsai
           else inverse_rule_of_mixtures(fibre.G, matrix.G, vf))
    rho = (rule_of_mixtures(fibre.rho, matrix.rho, vf)
           if fibre.rho is not None and matrix.rho is not None else None)
    return Result(vf, method(xi_E2), method(xi_G12), E1, E2, nu12, nu21,
                  nu23, G12, E2 / (2 * (1 + nu23)), K, rho)


def compliance_matrix_3D(r: Result) -> list[list[float]]:
    S = [[0.] * 6 for _ in range(6)]
    S[0][0], S[1][1], S[2][2] = 1 / r.E1, 1 / r.E2, 1 / r.E2
    S[0][1] = S[1][0] = S[0][2] = S[2][0] = -r.nu12 / r.E1
    S[1][2] = S[2][1] = -r.nu23 / r.E2
    S[3][3], S[4][4], S[5][5] = 1 / r.G23, 1 / r.G12, 1 / r.G12
    return S


def compliance_matrix_2D(r: Result) -> list[list[float]]:
    return [[1 / r.E1, -r.nu12 / r.E1, 0.],
            [-r.nu12 / r.E1, 1 / r.E2, 0.], [0., 0., 1 / r.G12]]


def reduced_stiffness_2D(r: Result) -> list[list[float]]:
    d = 1 - r.nu12 * r.nu21
    q11, q22, q12 = r.E1 / d, r.E2 / d, r.nu12 * r.E2 / d
    return [[q11, q12, 0.], [q12, q22, 0.], [0., 0., r.G12]]


def _print_matrix(matrix: list, title: str, labels: list[str]) -> None:
    width = 12
    print(f"\n{title}\n{'':>7}" + "".join(f"{x:>{width}}" for x in labels))
    for label, row in zip(labels, matrix):
        print(f"{label:>7}" + "".join(f"{x:{width}.3f}" for x in row))


def _print_values(rows: list[tuple[str, float, str]]) -> None:
    for label, value, unit in rows:
        print(f"  {label:<30} = {value:.3f}{' ' + unit if unit else ''}")


TARGETS = {
    "ALL": "Everything", "E1": "Longitudinal Young's modulus",
    "E2": "Transverse Young's modulus", "nu12": "Major Poisson's ratio",
    "nu21": "Minor Poisson's ratio", "nu23": "Transverse Poisson's ratio",
    "K": "Bulk modulus", "G12": "In-plane shear modulus",
    "G23": "Transverse shear modulus", "rho": "Density",
    "S3D": "3D compliance matrix", "S2D": "Plane-stress compliance matrix",
    "Q2D": "Plane-stress stiffness matrix"}

NEEDS = {
    "E1": "E", "E2": "E xE", "nu12": "nu", "nu21": "E nu xE",
    "nu23": "E nu xE", "K": "E nu", "G12": "E nu xG",
    "G23": "E nu xE", "rho": "rho", "S3D": "E nu xE xG",
    "S2D": "E nu xE xG", "Q2D": "E nu xE xG",
    "ALL": "E nu rho xE xG"}


def _ask(prompt: str, default: float | None = None) -> float:
    while True:
        raw = input(f"{prompt}{f' [{default:g}]' if default is not None else ''}: ").strip()
        try:
            return default if not raw and default is not None else float(raw.replace(",", "."))
        except (ValueError, TypeError):
            print("Enter a number.")


def _selection(raw: str) -> list[str]:
    try:
        keys = list(dict.fromkeys(list(TARGETS)[int(x) - 1]
                                  for x in raw.replace(" ", "").split(",")))
        if any(int(x) < 1 for x in raw.replace(" ", "").split(",")):
            raise IndexError
    except (ValueError, IndexError):
        raise ValueError(f"Choose numbers from 1 to {len(TARGETS)}")
    return ["ALL"] if "ALL" in keys else keys


def _material(role: str, needed: set[str]) -> Material:
    E = _ask(f"{role} E") if "E" in needed else 1
    nu = _ask(f"{role} nu") if "nu" in needed else .3
    rho = (_ask(f"{role} rho (0 = skip)", 0) or None
           if "rho" in needed else None)
    return Material(E, nu, rho=rho, name=role.lower())


def _report_selected(keys: list[str], r: Result, unit: str) -> None:
    matrices = {
        "S3D": (r.compliance_3D(), f"S (3D compliance) [1/{unit}]", ["1", "2", "3", "4=23", "5=13", "6=12"]),
        "S2D": (r.compliance_2D(), f"S (plane stress) [1/{unit}]", ["1", "2", "6=12"]),
        "Q2D": (r.stiffness_2D(), f"Q (plane stress) [{unit}]", ["1", "2", "6=12"])}
    labels = {"E1": "E1", "E2": "E2 = E3", "nu12": "nu12 = nu13",
              "nu21": "nu21 = nu31", "nu23": "nu23 = nu32", "K": "K",
              "G12": "G12 = G13", "G23": "G23 = G32", "rho": "rho"}
    rows = []
    for key in keys:
        if key in matrices:
            _print_matrix(*matrices[key])
        else:
            value = getattr(r, key)
            if value is None:
                print(f"{labels[key]} unavailable: enter both densities")
            else:
                rows.append((labels[key], value,
                             "" if key.startswith("nu") or key == "rho" else unit))
    _print_values(rows)


def run_interactive() -> None:
    while True:
        print("\nMICROMECHANICS OF AN ALIGNED FIBRE COMPOSITE PLY")
        for i, (key, text) in enumerate(TARGETS.items(), 1):
            print(f"{i:>2}. {key:<4} - {text}")
        while True:
            try:
                keys = _selection(input("Selection (comma-separated): "))
                break
            except ValueError as error:
                print(error)
        needed = set(" ".join(NEEDS[k] for k in keys).split())
        fibre, matrix = _material("Fibre", needed), _material("Matrix", needed)
        vf = _ask("Fibre volume fraction")
        xi_E2 = _ask("xi for E2", 2) if "xE" in needed else 2
        xi_G12 = _ask("xi for G12", 1) if "xG" in needed else 1
        unit = input("Modulus unit [GPa]: ").strip() or "GPa"
        result = compute(fibre, matrix, vf, xi_E2=xi_E2, xi_G12=xi_G12)
        if keys == ["ALL"]:
            result.print_report(unit)
            result.print_matrices(unit)
        else:
            _report_selected(keys, result, unit)
        if input("Calculate something else? [y/N]: ").strip().lower() not in ("y", "yes"):
            break


def demo() -> None:
    result = compute(Material(76, .2, name="fibre"),
                     Material(4, .3, name="matrix"), .55)
    result.print_report()
    result.print_matrices()


if __name__ == "__main__":
    import sys
    try:
        demo() if "--demo" in sys.argv else run_interactive()
    except (KeyboardInterrupt, EOFError):
        print("\nBye.")
