using System;
using System.Collections.Generic;
using System.Linq;
using System.Web;

namespace ProyectoDeteccionBalas.Models
{
    public class ImpactosBala
    {
        public int IdImpacto { get; set; }
        public int IdUsuario { get; set; }
        public DateTime Fecha { get; set; }
        public string Ubicacion { get; set; }
        public float Precision { get; set; }
        public string RutaImagen { get; set; }

    }
}