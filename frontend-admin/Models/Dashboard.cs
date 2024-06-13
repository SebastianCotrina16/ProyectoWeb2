using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace ProyectoDeteccionBalas.Models
{
    public class Dashboard
    {
        public int TotalUsuarios { get; set; }
        public int TotalImpactos { get; set; }
        public int TotalReportes { get; set; }
        public double PromedioPrecision { get; set; }
    }
}