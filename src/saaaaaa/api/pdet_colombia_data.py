"""
PDET Colombia Data - Complete Dataset of 170 Municipalities across 16 Subregions

This module contains the authoritative dataset of all Colombian PDET (Programas de
Desarrollo con Enfoque Territorial) municipalities organized by subregion and department.

Data Source: Colombian Government - Agencia de Renovación del Territorio (ART)
Total Subregions: 16
Total Municipalities: 170

Author: AtroZ Dashboard Integration Team
Version: 1.0.0
Last Updated: 2024-11-14
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class Municipality:
    """Represents a single PDET municipality"""
    name: str
    department: str
    subregion: str
    subregion_id: str
    population: Optional[int] = None
    area_km2: Optional[float] = None
    metadata: Dict[str, any] = field(default_factory=dict)


@dataclass
class PDETSubregion:
    """Represents a PDET subregion with its municipalities"""
    id: str
    name: str
    departments: List[str]
    municipalities: List[str]
    municipality_count: int
    description: Optional[str] = None


# Complete PDET Dataset - 16 Subregions, 170 Municipalities
PDET_SUBREGIONS = [
    PDETSubregion(
        id="alto-patia",
        name="Alto Patía y Norte del Cauca",
        departments=["Cauca", "Valle del Cauca"],
        municipalities=[
            "Buenos Aires", "Suárez", "Morales", "Caloto",
            "Corinto", "Toribio", "Jambaló", "Caldono",
            "Santander de Quilichao", "Miranda", "Puerto Tejada",
            "Villa Rica", "Jamundí", "Cajibaí",
            "Argelia", "El Tambo", "Patía", "Mercaderes",
            "Florencia", "Balboa", "Bolívar", "Sucre",
            "Timbío", "Rosas"
        ],
        municipality_count=24,
        description="Subregión del suroccidente colombiano con alta diversidad étnica"
    ),
    PDETSubregion(
        id="arauca",
        name="Arauca",
        departments=["Arauca"],
        municipalities=["Arauca", "Arauquita", "Saravena", "Tame"],
        municipality_count=4,
        description="Frontera con Venezuela, zona de producción petrolera"
    ),
    PDETSubregion(
        id="bajo-cauca",
        name="Bajo Cauca y Nordeste Antioqueño",
        departments=["Antioquia"],
        municipalities=[
            "Cáceres", "Tarazá", "Caucasia", "El Bagre",
            "Nechí", "Zaragoza", "Anorí", "Amalfi",
            "Remedios", "Segovia", "Vegachí", "Yalí", "Yondó"
        ],
        municipality_count=13,
        description="Zona minera aurífera del norte de Antioquia"
    ),
    PDETSubregion(
        id="catatumbo",
        name="Catatumbo",
        departments=["Norte de Santander"],
        municipalities=[
            "Convención", "El Carmen", "El Tarra", "Hacarí",
            "San Calixto", "Sardinata", "Teorama", "Tibú",
            "Cúcuta", "Puerto Santander", "Villa del Rosario"
        ],
        municipality_count=11,
        description="Frontera con Venezuela, histórica zona de conflicto"
    ),
    PDETSubregion(
        id="choco",
        name="Chocó",
        departments=["Chocó"],
        municipalities=[
            "Alto Baudó", "Bagadó", "Bajo Baudó", "Bojayá",
            "Carmen del Darién", "Condoto", "Istmina", "Litoral de San Juan",
            "Medio Atrato", "Medio Baudó", "Medio San Juan",
            "Quibdó", "Río Quito", "Unguía"
        ],
        municipality_count=14,
        description="Pacífico chocoano con alta diversidad biológica y cultural"
    ),
    PDETSubregion(
        id="caguan",
        name="Cuenca del Caguán y Piedemonte Caqueteño",
        departments=["Caquetá", "Meta"],
        municipalities=[
            "Albania", "Belén de los Andaquíes", "Cartagena del Chairá",
            "Curillo", "El Doncello", "El Paujil", "Florencia",
            "Milán", "Montañita", "Morelia", "Puerto Rico",
            "San José del Fragua", "San Vicente del Caguán",
            "Solano"
        ],
        municipality_count=14,
        description="Amazonía colombiana, transición andino-amazónica"
    ),
    PDETSubregion(
        id="macarena",
        name="Macarena-Guaviare",
        departments=["Meta", "Guaviare"],
        municipalities=[
            "La Macarena", "Mesetas", "Puerto Concordia",
            "Puerto Lleras", "Puerto Rico", "San Juan de Arama",
            "Uribe", "Vista Hermosa", "El Retorno", "Calamar"
        ],
        municipality_count=10,
        description="Zona de conservación del Parque Nacional Natural Sierra de La Macarena"
    ),
    PDETSubregion(
        id="montes-maria",
        name="Montes de María",
        departments=["Bolívar", "Sucre"],
        municipalities=[
            "Carmen de Bolívar", "El Guamo", "María la Baja",
            "San Jacinto", "San Juan Nepomuceno", "Zambrano",
            "Córdoba", "El Salado", "Chalán", "Colosó",
            "Morroa", "Ovejas", "San Onofre", "Tolú Viejo",
            "Sampués"
        ],
        municipality_count=15,
        description="Zona de montañas y valles entre la Costa Caribe y el interior"
    ),
    PDETSubregion(
        id="pacifico-medio",
        name="Pacífico Medio",
        departments=["Valle del Cauca", "Cauca"],
        municipalities=["Buenaventura", "Dagua", "La Cumbre", "Calima", "Guapi", "López de Micay", "Timbiquí"],
        municipality_count=7,
        description="Principal puerto sobre el Océano Pacífico y costa pacífica caucana"
    ),
    PDETSubregion(
        id="pacifico-narinense",
        name="Pacífico y Frontera Nariñense",
        departments=["Nariño"],
        municipalities=[
            "Barbacoas", "El Charco", "Francisco Pizarro",
            "La Tola", "Magüí Payán", "Mosquera",
            "Olaya Herrera", "Ricaurte", "Roberto Payán",
            "Santa Bárbara", "Tumaco"
        ],
        municipality_count=11,
        description="Costa pacífica nariñense, frontera con Ecuador"
    ),
    PDETSubregion(
        id="putumayo",
        name="Putumayo",
        departments=["Putumayo"],
        municipalities=[
            "Mocoa", "Orito", "Puerto Asís", "Puerto Caicedo",
            "Puerto Guzmán", "Puerto Leguízamo", "San Francisco",
            "San Miguel", "Santiago", "Sibundoy", "Valle del Guamuez"
        ],
        municipality_count=11,
        description="Piedemonte amazónico, frontera con Ecuador y Perú"
    ),
    PDETSubregion(
        id="sierra-nevada",
        name="Sierra Nevada-Perijá-Zona Bananera",
        departments=["Cesar", "La Guajira", "Magdalena"],
        municipalities=[
            "Pueblo Bello", "Valledupar", "Agustín Codazzi",
            "La Paz", "Manaure Balcón del Cesar", "San Diego",
            "Ciénaga", "Fundación", "Zona Bananera", "Dibulla"
        ],
        municipality_count=10,
        description="Macizo montañoso independiente más alto del mundo sobre el mar"
    ),
    PDETSubregion(
        id="sur-bolivar",
        name="Sur de Bolívar",
        departments=["Bolívar"],
        municipalities=[
            "Arenal", "Cantagallo", "Morales", "Norosí",
            "Río Viejo", "San Pablo", "Santa Rosa del Sur"
        ],
        municipality_count=7,
        description="Zona minera del sur del departamento de Bolívar"
    ),
    PDETSubregion(
        id="sur-cordoba",
        name="Sur de Córdoba",
        departments=["Córdoba"],
        municipalities=[
            "Montelíbano", "Puerto Libertador", "San José de Uré",
            "Tierralta", "Valencia"
        ],
        municipality_count=5,
        description="Zona del alto Sinú con resguardos indígenas"
    ),
    PDETSubregion(
        id="sur-tolima",
        name="Sur del Tolima",
        departments=["Tolima"],
        municipalities=["Ataco", "Chaparral", "Planadas", "Rioblanco"],
        municipality_count=4,
        description="Zona montañosa del sur del Tolima"
    ),
    PDETSubregion(
        id="uraba",
        name="Urabá Antioqueño",
        departments=["Antioquia"],
        municipalities=[
            "Apartadó", "Arboletes", "Carepa", "Chigorodó",
            "Mutatá", "Necoclí", "San Juan de Urabá", "San Pedro de Urabá",
            "Turbo", "Vigía del Fuerte"
        ],
        municipality_count=10,
        description="Zona bananera y ganadera del Caribe antioqueño"
    )
]


# Flattened list of all municipalities with full metadata
ALL_MUNICIPALITIES = []
for subregion in PDET_SUBREGIONS:
    for muni_name in subregion.municipalities:
        # Determine primary department (first in list for multi-department subregions)
        primary_dept = subregion.departments[0]
        
        ALL_MUNICIPALITIES.append(
            Municipality(
                name=muni_name,
                department=primary_dept,
                subregion=subregion.name,
                subregion_id=subregion.id,
                metadata={
                    "all_departments": subregion.departments,
                    "subregion_description": subregion.description
                }
            )
        )


# Utility functions
def get_subregion_by_id(subregion_id: str) -> Optional[PDETSubregion]:
    """Get a PDET subregion by its ID"""
    for subregion in PDET_SUBREGIONS:
        if subregion.id == subregion_id:
            return subregion
    return None


def get_municipalities_by_subregion(subregion_id: str) -> List[Municipality]:
    """Get all municipalities in a specific subregion"""
    return [m for m in ALL_MUNICIPALITIES if m.subregion_id == subregion_id]


def get_municipalities_by_department(department: str) -> List[Municipality]:
    """Get all municipalities in a specific department"""
    return [m for m in ALL_MUNICIPALITIES if m.department == department]


def get_all_departments() -> List[str]:
    """Get list of all departments with PDET municipalities"""
    departments = set()
    for subregion in PDET_SUBREGIONS:
        departments.update(subregion.departments)
    return sorted(list(departments))


def get_statistics() -> Dict[str, any]:
    """Get dataset statistics"""
    return {
        "total_subregions": len(PDET_SUBREGIONS),
        "total_municipalities": len(ALL_MUNICIPALITIES),
        "total_departments": len(get_all_departments()),
        "departments": get_all_departments(),
        "subregions_by_municipality_count": sorted(
            [(s.name, s.municipality_count) for s in PDET_SUBREGIONS],
            key=lambda x: x[1],
            reverse=True
        )
    }


# Module-level validation
assert len(ALL_MUNICIPALITIES) == sum(s.municipality_count for s in PDET_SUBREGIONS), \
    "Municipality count mismatch between individual counts and total list"

assert len(PDET_SUBREGIONS) == 16, \
    f"Expected 16 PDET subregions, found {len(PDET_SUBREGIONS)}"

# Print statistics when module is imported
if __name__ == "__main__":
    stats = get_statistics()
    print("=" * 60)
    print("PDET COLOMBIA DATASET STATISTICS")
    print("=" * 60)
    print(f"Total Subregions: {stats['total_subregions']}")
    print(f"Total Municipalities: {stats['total_municipalities']}")
    print(f"Total Departments: {stats['total_departments']}")
    print("\nDepartments:")
    for dept in stats['departments']:
        munis = get_municipalities_by_department(dept)
        print(f"  - {dept}: {len(munis)} municipalities")
    print("\nSubregions (by municipality count):")
    for name, count in stats['subregions_by_municipality_count']:
        print(f"  - {name}: {count} municipalities")
    print("=" * 60)
