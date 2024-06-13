using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace ProyectoDeteccionBalas.Models
{
    public class Reporte
    {
        public int IdReporte { get; set; }
        public int IdUsuario { get; set; }
        public DateTime FechaReporte { get; set; }
        public int TotalImpactos { get; set; }
        public float PromedioPrecision { get; set; }
        public string Detalles { get; set; }
    }
}